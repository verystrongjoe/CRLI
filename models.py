import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.rnn_hid_size = config.G_hiddensize
        self.seq_len = self.config.seq_len
        self.input_dim = self.config.input_dim
        self.device = torch.device(f'cuda:{config.gpu_num}' if torch.cuda.is_available() else 'cpu')

        # https://towardsdatascience.com/from-a-lstm-cell-to-a-multilayer-lstm-network-with-pytorch-2899eb5696f3
        self.fwd_rnn_cell = nn.ModuleList(
             [
                nn.LSTMCell((self.input_dim if layer==0 else self.rnn_hid_size), self.rnn_hid_size)
                for layer in range(config.G_layer)
             ]
        )

        self.bwd_rnn_cell = nn.ModuleList(
             [
                nn.LSTMCell((self.input_dim if layer==0 else self.rnn_hid_size), self.rnn_hid_size)
                for layer in range(config.G_layer)
             ]
        )

        # self.fwd_rnn_cell = nn.LSTMCell(self.input_dim, self.rnn_hid_size)
        # self.bwd_rnn_cell = nn.LSTMCell(self.input_dim, self.rnn_hid_size)

        self.fc_begin_value = nn.Linear(1, self.input_dim)
        self.fc_end_value = nn.Linear(1, self.input_dim)

        self.out = nn.Linear(self.rnn_hid_size, self.input_dim)
        # self.out = nn.Linear(self.rnn_hid_size, self.input_dim).to(self.device)

    def forward(self, data):
        fwd_values = data['values']
        fwd_masks = data['masks']
        batch_size = fwd_values.size()[0]

        assert fwd_values.shape == (batch_size, self.config.seq_len, self.input_dim)
        assert fwd_masks.shape == (batch_size, self.config.seq_len, self.input_dim)

        bwd_values = torch.flip(fwd_values, (1,))
        bwd_masks = torch.flip(fwd_masks, (1,))

        fwd_start_value = Variable(torch.ones(batch_size, 1), requires_grad=False).to(self.device) * 128
        bwd_start_value = Variable(torch.ones(batch_size, 1), requires_grad=False).to(self.device) * -128

        fwd_start = self.fc_begin_value(fwd_start_value)
        bwd_start = self.fc_end_value(bwd_start_value)

        h_fwds, c_fwds = [], []
        h_bwds, c_bwds = [], []

        for layer in range(self.config.G_layer):
            h_fwds.append(Variable(torch.zeros((fwd_values.size()[0], self.rnn_hid_size))))   # (B, H)
            c_fwds.append(Variable(torch.zeros((fwd_values.size()[0], self.rnn_hid_size))))  # (B, H)

            h_bwds.append(Variable(torch.zeros((fwd_values.size()[0], self.rnn_hid_size))))   # (B, H)
            c_bwds.append(Variable(torch.zeros((fwd_values.size()[0], self.rnn_hid_size))))  # (B, H)

        # append first identifier to data (B, T, 1)
        # for the first time, we set first value as for equation(5).
        for layer in range(1, self.config.G_layer):
            if layer == 0:
                h_fwds[layer], c_fwds[layer] = self.fwd_rnn_cell[layer](fwd_start, (h_fwds[layer], c_fwds[layer]))
                h_bwds[layer], c_bwds[layer] = self.bwd_rnn_cell[layer](bwd_start, (h_bwds[layer], c_bwds[layer]))
            else:
                h_fwds[layer], c_fwds[layer] = self.fwd_rnn_cell[layer](h_fwds[layer-1], (h_fwds[layer], c_fwds[layer]))
                h_bwds[layer], c_bwds[layer] = self.bwd_rnn_cell[layer](h_bwds[layer-1], (h_bwds[layer], c_bwds[layer]))

        imputations = []

        for t in range(self.seq_len):
            x_fwd = fwd_values[:, t, :]
            m_fwd = fwd_masks[:, t, :]

            x_bwd = bwd_values[:, t, :]
            m_bwd = bwd_masks[:, t, :]

            x_imputed_fwd = self.out(h_fwds[self.config.G_layer-1])
            x_imputed_bwd = self.out(h_bwds[self.config.G_layer-1])

            c_c_fwd = (1-m_fwd) * x_imputed_fwd + m_fwd * x_fwd
            c_c_bwd = (1-m_bwd) * x_imputed_bwd + m_bwd * x_bwd

            for layer in range(self.config.G_layer):
                if layer == 0:
                    h_fwds[layer], c_fwds[layer] = self.fwd_rnn_cell[layer](c_c_fwd, (h_fwds[layer], c_fwds[layer]))
                    h_bwds[layer], c_bwds[layer] = self.bwd_rnn_cell[layer](c_c_bwd, (h_bwds[layer], c_bwds[layer]))
                else:
                    h_fwds[layer], c_fwds[layer] = self.fwd_rnn_cell[layer](h_fwds[layer - 1], (h_fwds[layer], c_fwds[layer]))
                    h_bwds[layer], c_bwds[layer] = self.bwd_rnn_cell[layer](h_bwds[layer - 1], (h_bwds[layer], c_bwds[layer]))

            h = (h_fwds[self.config.G_layer-1] +
                 torch.flip(h_bwds[self.config.G_layer-1], (1,))
                 )/2

            if t == (self.seq_len - 1):
                imputations.append(h)

        imputations = torch.cat(imputations, dim=1).reshape(batch_size, self.config.seq_len, self.input_dim)

        # 원래 논문에는 1이 존재하는 데이터 0이 missing data임을 감안

        return h, imputations, imputations * (1 - fwd_masks) + fwd_values * fwd_masks


class Decoder(nn.Module):
    def __init__(self, config,  rnn_hid_size, output_dim):
        super(Decoder, self).__init__()
        self.config = config
        self.latent_dim = rnn_hid_size

        self.rnn_cell = nn.LSTMCell(rnn_hid_size, rnn_hid_size)
        self.rnn_output = nn.Linear(rnn_hid_size, output_dim)

        self.rnn_hid_size = rnn_hid_size
        self.input_dim = self.config.input_dim
        self.seq_len = config.seq_len

    def forward(self, x):  # bi-rnn -> forward states + backward states
        batch_size = x.size()[0]

        # todo : 여기 start value를 어떻게 정의해야하나?
        start_value = Variable(torch.ones(batch_size, self.rnn_hid_size), requires_grad=False) * 128
        start_value = start_value

        h = Variable(torch.zeros((batch_size, self.rnn_hid_size))) # (B, H)
        c = Variable(torch.zeros((batch_size, self.rnn_hid_size))) # (B, H)

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        h, c = self.rnn_cell(start_value, (h, c))
        outputs = []

        for t in range(self.seq_len):
            h, c = self.rnn_cell(c, (h, c))
            outputs.append(self.rnn_output(h))
        return torch.stack(outputs, dim=1)


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        self.output_layer = nn.Linear(32, self.config.input_dim)
        self.disc_cells = nn.ModuleList(
            [nn.LSTMCell(input_size=i, hidden_size=h)for (i, h) in
             zip([self.config.input_dim, 32, 16, 8, 16], [32, 16, 8, 16, 32])
             ])

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
        latent_dim = config.G_hiddensize
        output_dim = config.input_dim

        super(CRLI, self).__init__()
        self.config = config

        # start branch
        self.generator = Generator(config=config)

        # upper branch
        self.discriminator = Discriminator(self.config)

        # lower branch
        self.fully_connected = nn.Linear(rnn_hid_size, latent_dim)
        self.decoder = Decoder(config, rnn_hid_size, output_dim)

        # spectral relaxation for k-means clustering
        # F is a cluster indicator matrix with shape of N x k (k is number of cluster in this dataset)
        # learning of H is dynamic instead of static ->  learning is done iteratively.
        # Ky fan theorem, F can be obtained by computing the k-truncated singular value decomposition of H
        f = torch.empty(config.batch_size, config.k_cluster, dtype=torch.float32)
        torch.nn.init.orthogonal_(f, gain=1)  # by equation 9  F^TxF = I
        self.F = torch.autograd.Variable(f, requires_grad=False)

    def forward(self, x):
        h, impute, imputed = self.generator(x)
        disc_output = self.discriminator(imputed)
        latent = self.fully_connected(h)
        reconstructed = self.decoder(latent)
        return disc_output, impute, latent, reconstructed


    def update_F(self, F_new):
        #  assign a new value to a pytorch Variable without breaking backpropagation!!
        #  thanks to https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
        self.F.data = F_new.data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--T_update_F', type=int, required=False, default=1)
    parser.add_argument('--G_steps', type=int, required=False, default=1)
    parser.add_argument('--D_steps', type=int, required=False, default=1)
    parser.add_argument('--epoch', type=int, required=False, default=30)
    parser.add_argument('--G_layer', type=int, required=False, default=1)
    parser.add_argument('--IsD', type=bool, required=False, default=False)
    parser.add_argument('--cell_type', type=str, required=False, default='gru')
    parser.add_argument('--data_dir', type=str, required=False, default='dataset')
    parser.add_argument('--gpu_num', type=int, required=False, default=1)
    parser.add_argument('--seq_len', type=int, required=False, default=48)
    parser.add_argument('--learning_rate', type=float, required=False, default=5e-3)
    parser.add_argument('--lambda_kmeans', type=float, required=False, default=1e-3)
    parser.add_argument('--input_dim', type=int, required=False, default=1)
    parser.add_argument('--G_hiddensize', type=int, required=False, default=20 * 10)
    parser.add_argument('--k_cluster', type=int, required=False, default=30)

    config = parser.parse_args()
    device = torch.device(f'cuda:{config.gpu_num}' if torch.cuda.is_available() else 'cpu')
    m = CRLI(config)
    m.to(device)
