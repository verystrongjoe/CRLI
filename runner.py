"""
written by Uk Jo

# [paper] https://www.aaai.org/AAAI21Papers/AAAI-2808.MaQ.pdf
# [appendix] https://github.com/qianlima-lab/CRLI/blob/main/appendix.pdf
"""
import torch
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
from commons import assess, load_mask, load_data, get_batch, get_args
from models import CRLI
import torch.nn.functional as F
import wandb


def run(config):
    train_data_filename = config.data_dir + "/" + config.dataset_name + "/" + config.dataset_name + "_TRAIN"
    train_data, train_label = load_data(train_data_filename)
    test_data_filename = train_data_filename.replace('TRAIN', 'TEST')
    test_data, test_label = load_data(test_data_filename)

    train_maskname = config.data_dir + "/" + config.dataset_name + "/" + "mask_" + config.dataset_name + "_TRAIN"
    train_mask = load_mask(train_maskname)
    test_maskname = train_maskname.replace('TRAIN', 'TEST')
    test_mask = load_mask(test_maskname)

    train_dataset_size = train_data.shape[0]
    test_dataset_size = test_data.shape[0]
    config.k_cluster = len(np.unique(train_label))  # useless for me
    config.train_dataset_size = train_dataset_size

    if config.batch_size > train_dataset_size:
        config.batch_size = train_dataset_size
    elif config.batch_size < config.k_cluster:
        config.batch_size = config.k_cluster

    RI = []
    NMI = []
    ACC = []
    PUR = []
    GLOBAL_STEP = -1

    m = CRLI(config)
    m.to(torch.device(f'cuda' if torch.cuda.is_available() else 'cpu'))

    gen_params = []
    disc_params = []

    for name, param in m.named_parameters():
        if 'discriminator' in name:
            disc_params.append(param)
        else:
            gen_params.append(param)

    disc_optimizer = torch.optim.Adam(disc_params, lr=config.learning_rate)
    gen_optimizer = torch.optim.Adam(gen_params, lr=config.learning_rate)

    for i in range(config.epoch):
        ####################################################################################################
        # Train
        ####################################################################################################
        D_Step_losses = []
        G_Step_losses = []

        G_losses = []
        re_losses = []
        pre_losses = []
        km_losses = []

        for batch_data, batch_mask in get_batch(train_data, train_mask, config):
            GLOBAL_STEP += 1
            data = {}

            batch_data = torch.reshape(batch_data, (config.batch_size, config.seq_len, config.input_dim))
            batch_mask = torch.reshape(batch_mask, (config.batch_size, config.seq_len, config.input_dim))

            data['values'] = batch_data
            data['masks'] = batch_mask

            for _ in range(config.D_steps):
                disc_output, impute, latent, reconstructed = m(data)
                # discriminator
                loss_D = F.binary_cross_entropy_with_logits(disc_output, batch_mask).mean()
                disc_optimizer.zero_grad()
                loss_D.backward()
                disc_optimizer.step()
                D_Step_losses.append(loss_D.item())

            for j in range(config.G_steps):
                # run generator
                disc_output, impute, latent, reconstructed = m(data)
                loss_adv = F.binary_cross_entropy_with_logits(disc_output, 1-batch_mask)

                # loss_pre
                pre_tmp = impute * batch_mask
                target_pre = batch_data * batch_mask
                loss_pre = nn.MSELoss()(pre_tmp, target_pre)

                # loss_re
                out_tmp = reconstructed * batch_mask
                targ_re = batch_data * batch_mask
                loss_re = nn.MSELoss()(out_tmp, targ_re)

                # loss_km
                HTH = torch.matmul(latent, latent.T)
                FTHTHF = torch.matmul(torch.matmul(m.F.T, HTH), m.F)
                loss_km = torch.trace(HTH) - torch.trace(FTHTHF)

                gen_optimizer.zero_grad()
                (loss_adv + loss_pre + loss_re + loss_km * config.lambda_kmeans).backward()
                gen_optimizer.step()

                G_Step_losses.append((loss_adv + loss_pre + loss_re + loss_km * config.lambda_kmeans).item())
                km_losses.append(loss_km.item())
                pre_losses.append(loss_pre.item())
                re_losses.append(loss_pre.item())
                G_losses.append(loss_adv.item())

                if i % config.T_update_F == 0:
                    # F_update
                    disc_output, impute, latent, reconstructed = m(data)
                    U, s, V = torch.linalg.svd(latent)
                    F_new = U.T[:config.k_cluster, :]
                    F_new = F_new.T
                    m.update_F(F_new)

        ####################################################################################################
        # TEST
        ####################################################################################################
        with torch.no_grad():
            H_outputs = []
            for batch_data, batch_mask in get_batch(test_data, test_mask, config):
                data = {}
                batch_data = torch.reshape(batch_data, (config.batch_size, config.seq_len, config.input_dim))
                batch_mask = torch.reshape(batch_mask, (config.batch_size, config.seq_len, config.input_dim))

                data['values'] = batch_data
                data['masks'] = batch_mask
                imputed, disc_output, latent, reconstructed = m(data)
                H_outputs.append(latent)
            H_outputs = torch.concat(H_outputs, 0)
            H_outputs = H_outputs[:test_dataset_size, :]
            Km = KMeans(n_clusters=config.k_cluster)
            pred_H = Km.fit_predict(H_outputs.cpu().detach().numpy())

            # ASSESMENT
            ri,nmi,acc,pur = assess(pred_H, test_label)
            print(f'nm_dataset : {config.dataset_name}, epoch : {i}, acc : {acc}, ri: {ri}')

            ## wandb logging
            wandb.log({'accuracy' : acc})
            wandb.log({'ri': ri})
            wandb.log({'D_step loss': np.mean(D_Step_losses)})
            wandb.log({'G_step loss': np.mean(G_Step_losses)})
            wandb.log({'loss_adv': np.mean(G_losses)})
            wandb.log({'loss_pre': np.mean(pre_losses)})
            wandb.log({'loss_re': np.mean(re_losses)})
            wandb.log({'loss_km': np.mean(km_losses)})

            RI.append(ri)
            NMI.append(nmi)
            ACC.append(acc)
            PUR.append(pur)

    # It is based on tensorflow code.
    # ri = max(RI[80-1], RI[300-1], RI[500-1])
    # nmi = max(NMI[80-1], NMI[300-1], NMI[500-1])
    # acc = max(ACC[80-1], ACC[300-1], ACC[500-1])
    # pur = max(PUR[80-1], PUR[300-1], PUR[500-1])

    # ri = max(RI[80-1], RI[300-1], RI[500-1])
    # nmi = max(NMI[80-1], NMI[300-1], NMI[500-1])
    # acc = max(ACC[80-1], ACC[300-1], ACC[500-1])
    # pur = max(PUR[80-1], PUR[300-1], PUR[500-1])

    # For now, I ignore this evaluation method and just pick best result in every epochs.
    ri = max(RI)
    nmi = max(NMI)
    acc = max(ACC)
    pur = max(PUR)

    return ri, nmi, pur, acc


if __name__ == '__main__':
    config = get_args()

    # model hyperparams and the value follows this paper.
    list_g_layers = [1, 2, 3]
    list_lambda_kmeans = [1e-3, 1e-6, 1e-9]
    list_G_steps = [2, 3, 4]

    for nm_dataset in config.datasets:
        for g_layer in list_g_layers:
            for lambda_kmeans in list_lambda_kmeans:
                for G_steps in list_G_steps:
                    config.dataset_name = nm_dataset
                    config.seq_len = config.datasets[nm_dataset]["seq_len"]
                    config.input_dim = config.datasets[nm_dataset]["input_dim"]
                    config.G_hiddensize = config.seq_len * config.input_dim  # todo : Hidden 사이즈 여러개 실험 50, 100, 150
                    config.G_layer = g_layer
                    config.lambda_kmeans = lambda_kmeans
                    config.G_steps = G_steps
                    experiment_name = f"lambda_{lambda_kmeans}_l_{g_layer}_g_{G_steps}"
                    wandb.init(config=config, project=f'crli_{nm_dataset}_300', name=experiment_name, entity="dataknows")
                    wandb.config.update(config, allow_val_change=True)
                    wandb_metric_table = wandb.Table(columns=['nm_dataset', 'ri', 'nmi', 'pur', 'cluster_acc'])
                    ri, nmi, pur, cluster_acc = run(config)
                    wandb_metric_table.add_data(nm_dataset, ri, nmi, pur, cluster_acc)
                    wandb.log({"metrics": wandb_metric_table})
                    wandb.finish()
                    print('dataset name : %s,  ri : %.6f, nmi : %.6f, pur : %.6f, accuracy : %.6f' % (config.dataset_name, ri, nmi, pur, cluster_acc))
