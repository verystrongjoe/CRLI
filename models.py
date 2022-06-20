import torch
import torch.nn as nn
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, config, input_dim=1):
        super(Generator, self).__init__()
        self.config = config
        self.rnn_hid_size = config.G_hiddensize
        self.seq_len = self.config.seq_len
        self.input_dim = input_dim
        self.device = torch.device(f'cuda:{config.gpu_num}' if torch.cuda.is_available() else 'cpu')

        self.fwd_rnn_cell = nn.LSTMCell(input_dim, self.rnn_hid_size).to(self.device)
        self.bwd_rnn_cell = nn.LSTMCell(input_dim, self.rnn_hid_size).to(self.device)
        self.impute_reg = nn.Linear(self.rnn_hid_size, self.input_dim).to(self.device)
        # self.out = nn.Linear(self.rnn_hid_size, self.input_dim).to(self.device)

    def forward(self, data):
        fwd_values = data['values']
        fwd_masks = data['masks']

        fwd_values = torch.unsqueeze(fwd_values, -1)
        fwd_masks = torch.unsqueeze(fwd_masks, -1)
        batch_size = fwd_values.size()[0]

        assert fwd_values.shape == (batch_size, self.config.seq_len, 1)

        bwd_values = torch.flip(fwd_values, (2,))
        bwd_masks = torch.flip(fwd_masks, (2,))

        fwd_start_value = Variable(torch.ones(batch_size, self.input_dim), requires_grad=False) * 128
        bwd_start_value = Variable(torch.ones(batch_size, self.input_dim), requires_grad=False) * -128
        fwd_start_value = fwd_start_value.to(self.device)
        bwd_start_value = bwd_start_value.to(self.device)

        h_fwd = Variable(torch.zeros((fwd_values.size()[0], self.rnn_hid_size)))  # (B, H)
        c_fwd = Variable(torch.zeros((fwd_values.size()[0], self.rnn_hid_size)))  # (B, H)

        h_bwd = Variable(torch.zeros((fwd_values.size()[0], self.rnn_hid_size)))  # (B, H)
        c_bwd = Variable(torch.zeros((fwd_values.size()[0], self.rnn_hid_size)))  # (B, H)

        if torch.cuda.is_available():
            h_fwd, c_fwd, h_bwd, c_bwd = h_fwd.cuda(), c_fwd.cuda(), h_bwd.cuda(), c_bwd.cuda()

        # append first identifier to data (B, T, 1)
        # for the first time, we set first value as for equation(5).
        h_fwd, c_fwd = self.fwd_rnn_cell(fwd_start_value, (h_fwd, c_fwd))
        h_bwd, c_bwd = self.bwd_rnn_cell(bwd_start_value, (h_bwd, c_bwd))
        imputations = []

        for t in range(self.seq_len):
            x_fwd = fwd_values[:, t, :]
            m_fwd = fwd_masks[:, t, :]

            x_bwd = bwd_values[:, t, :]
            m_bwd = bwd_masks[:, t, :]

            x_imputed_fwd = self.impute_reg(h_fwd)
            x_imputed_bwd = self.impute_reg(h_bwd)

            c_c_fwd = (1-m_fwd) * x_fwd + m_fwd * x_imputed_fwd
            c_c_bwd = (1-m_bwd) * x_bwd + m_bwd * x_imputed_bwd

            h_fwd, c_fwd = self.fwd_rnn_cell(c_c_fwd, (h_fwd, c_fwd))
            h_bwd, c_bwd = self.bwd_rnn_cell(c_c_bwd, (h_bwd, c_bwd))
            h = h_fwd + h_bwd

            if t == (self.seq_len - 1):
                imputations.append(h)

        imputations = torch.cat(imputations, dim=1).reshape(batch_size, self.config.seq_len, self.input_dim)

        # 원래 논문에는 1이 존재하는 데이터 0이 missing data임을 감안

        return h, imputations * (1 - fwd_masks) + fwd_values *  fwd_masks


class Decoder(nn.Module):
    def __init__(self, config,  latent_dim, rnn_hid_size, output_dim, input_dim=1):
        super(Decoder, self).__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.device = torch.device(f'cuda:{config.gpu_num}' if torch.cuda.is_available() else 'cpu')

        self.rnn_cell = nn.LSTMCell(latent_dim, rnn_hid_size).to(self.device)
        self.rnn_output = nn.Linear(rnn_hid_size, output_dim).to(self.device)

        self.rnn_hid_size = rnn_hid_size
        self.input_dim = input_dim
        self.seq_len = config.seq_len

    def forward(self, x):  # bi-rnn -> forward states + backward states
        batch_size = x.size()[0]

        # todo : 여기 start value를 어떻게 정의해야하나?
        start_value = Variable(torch.ones(batch_size, self.latent_dim), requires_grad=False) * 128
        start_value = start_value.to(self.device)

        h = Variable(torch.zeros((batch_size, self.rnn_hid_size))).to(self.device)  # (B, H)
        c = Variable(torch.zeros((batch_size, self.rnn_hid_size))) .to(self.device) # (B, H)

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        h, c = self.rnn_cell(start_value, (h, c))
        outputs = []

        for t in range(self.seq_len):
            h, c = self.rnn_cell(c, (h, c))
            outputs.append(self.rnn_output(h))
        return torch.stack(outputs, dim=1)


class Discriminator(nn.Module):
    def __init__(self, config, input_dim=1):
        super(Discriminator, self).__init__()
        self.config = config
        self.device = torch.device(f'cuda:{self.config.gpu_num}' if torch.cuda.is_available() else 'cpu')
        self.output_layer = nn.Linear(32, input_dim).to(self.device)
        self.disc_cells = nn.ModuleList(
            [nn.LSTMCell(input_size=i, hidden_size=h)for (i, h) in
             zip([input_dim, 32, 16, 8, 16], [32, 16, 8, 16, 32])
             ])
        self.disc_cells.to(self.device)

    def forward(self, x):
        outputs = []
        # B, T, 1 -> B, T, H_out
        for t in range(x.shape[1]):  # time seq
            input = x[:, t, :]  # b, t, h    첫번째 b, h  32, 1
            for i, rnn_cell in enumerate(self.disc_cells):
                if i == 0:
                    (h, c) = rnn_cell(input)
                else:
                    (h, c) = rnn_cell(h)
            outputs.append(h)
        return self.output_layer(torch.stack(outputs, dim=1))


class CRLI(nn.Module):
    def __init__(self, config):
        rnn_hid_size = config.G_hiddensize
        latent_dim = 16
        output_dim = 1

        super(CRLI, self).__init__()
        self.config = config
        self.device = torch.device(f'cuda:{config.gpu_num}' if torch.cuda.is_available() else 'cpu')

        # upper branch
        self.generator = Generator(config=config)
        self.discriminator = Discriminator(self.config)

        # lower branch
        self.fully_connected = nn.Linear(rnn_hid_size, latent_dim).to(self.device)
        self.decoder = Decoder(config, latent_dim, rnn_hid_size, output_dim).to(self.device)

        ## third
        f = torch.empty(config.batch_size, config.k_cluster, dtype=torch.float32)
        torch.nn.init.orthogonal_(f, gain=1)
        self.F = torch.autograd.Variable(f, requires_grad=False).to(self.device)


    def forward(self, x):
        h, imputed = self.generator(x)
        disc_output = self.discriminator(imputed)
        latent = self.fully_connected(h)
        reconstructed = self.decoder(latent)
        return imputed, disc_output, latent, reconstructed

    def update_F(self, F_new):
        #  assign a new value to a pytorch Variable without breaking backpropagation!!
        #  thanks to https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
        self.F.data = F_new.data


