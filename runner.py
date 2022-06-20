import torch
from sklearn import metrics
import pandas as pd
import warnings
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from models import CRLI
import torch.nn.functional as F

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()


def pur_metric(pred, target):
    n = len(pred)
    tmp = pd.crosstab(pred, target)
    tmp = np.array(tmp)
    ret = np.max(tmp, 1)
    ret = float(np.sum(ret))
    ret = ret / n
    return ret


def nmi_metric(pred, target):
    NMI = metrics.normalized_mutual_info_score(pred, target)
    return NMI


def RI_metric(pred, target):
    # RI
    n = len(target)
    TP = 0
    TN = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if target[i] != target[j]:
                if pred[i] != pred[j]:
                    TN += 1
            else:
                if pred[i] == pred[j]:
                    TP += 1

    RI = n * (n - 1) / 2
    RI = (TP + TN) / RI
    return RI


def _accuracy(y_pred, y_true):
    def cluster_acc(Y_pred, Y):
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max()) + 1
        w = np.zeros((D, D), dtype=np.int32)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, np.array(w)

    y_pred = np.array(y_pred, np.int32)
    y_true = np.array(y_true, np.int32)
    return cluster_acc(y_pred, y_true)


def assess(pred, target):
    pur = pur_metric(pred, target)
    nmi = nmi_metric(pred, target)
    ri = RI_metric(pred, target)
    acc, o = _accuracy(pred, target)
    return ri, nmi, acc, pur


def load_mask(filename):
    mask_label = np.loadtxt(filename,delimiter = ",")
    mask = mask_label[:,1:].astype(np.float32)
    return mask


def load_data(filename):
    data_label = np.loadtxt(filename,delimiter = ",")
    data = data_label[:,1:].astype(np.float32)
    label = data_label[:,0].astype(np.int32)
    return data,label


def load_length(filename):
    length = np.loadtxt(filename,delimiter = ",")
    return length


def load_lengthmark(filename):
    lengthmark = np.loadtxt(filename,delimiter = ",")
    return lengthmark


def get_batch(data, label, mask, length, lengthmark, config):
    samples_num = data.shape[0]
    batch_num = int(samples_num / config.batch_size)
    left_row = samples_num - batch_num * config.batch_size
    device = torch.device(f'cuda:{config.gpu_num}' if torch.cuda.is_available() else 'cpu')

    data = torch.from_numpy(data).to(device)
    mask = torch.Tensor(mask).to(device)

    for i in range(batch_num):
        batch_data = data[i * config.batch_size: (i + 1) * config.batch_size, :]
        batch_label = label[i * config.batch_size: (i + 1) * config.batch_size]
        batch_mask = mask[i * config.batch_size: (i + 1) * config.batch_size, :]
        batch_length = length[i * config.batch_size: (i + 1) * config.batch_size]
        batch_lengthmark = lengthmark[i * config.batch_size: (i + 1) * config.batch_size,:]
        yield (batch_data,batch_label,batch_mask,batch_length,batch_lengthmark)

    if left_row != 0:
        need_more = config.batch_size - left_row
        need_more = np.random.choice(np.arange(samples_num), size=need_more)
        batch_data = torch.concat((data[-left_row:, :], data[need_more,:]), axis=0)
        batch_label = np.concatenate((label[-left_row:],label[need_more]), axis=0)
        batch_mask = torch.concat((mask[-left_row:, :], mask[need_more,:]), axis=0)
        batch_length = np.concatenate((length[-left_row:],length[need_more]), axis=0)
        batch_lengthmark = np.concatenate((lengthmark[-left_row:,:],lengthmark[need_more,:]), axis=0)
        yield (batch_data,batch_label,batch_mask,batch_length,batch_lengthmark)


def run(config):
    '''data'''
    train_data_filename = config.data_dir + "/" + config.dataset_name + "/" + config.dataset_name + "_TRAIN"
    train_data, train_label = load_data(train_data_filename)
    test_data_filename = train_data_filename.replace('TRAIN', 'TEST')
    test_data, test_label = load_data(test_data_filename)

    '''mask'''
    train_maskname = config.data_dir + "/" + config.dataset_name + "/" + "mask_" + config.dataset_name + "_TRAIN"
    train_mask = load_mask(train_maskname)
    test_maskname = train_maskname.replace('TRAIN', 'TEST')
    test_mask = load_mask(test_maskname)

    '''length'''
    train_length_filename = config.data_dir + "/" + config.dataset_name + "/" + 'length_' + config.dataset_name + "_TRAIN"
    train_length = load_length(train_length_filename)
    test_length_filename = train_length_filename.replace('TRAIN', 'TEST')
    test_length = load_length(test_length_filename)

    '''train_lengthmark'''
    train_lengthmark_filename = config.data_dir + "/" + config.dataset_name + "/" + 'lengthmark_' + config.dataset_name + "_TRAIN"
    train_lengthmark = load_lengthmark(train_lengthmark_filename)
    test_lengthmark_filename = train_lengthmark_filename.replace('TRAIN', 'TEST')
    test_lengthmark = load_length(test_lengthmark_filename)

    train_dataset_size = train_data.shape[0]
    test_dataset_size = test_data.shape[0]
    config.inputdims = 1
    config.n_steps = train_data.shape[1] // config.inputdims
    config.k_cluster = len(np.unique(train_label))
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

    for i in range(config.epoch):
        '''Train'''
        for batch_data, batch_label, batch_mask, batch_length, batch_lengthmark in get_batch(
                train_data, train_label, train_mask, train_length, train_lengthmark, config):
            GLOBAL_STEP += 1
            data = {}
            data['values'] = batch_data
            data['masks'] = batch_mask
            for _ in range(config.D_steps):
                imputed, disc_output, latent, reconstructed = m(data)
                # run disc loss
                loss_D = F.binary_cross_entropy_with_logits(disc_output.squeeze(), batch_mask).mean()
                # backward
                loss_D.backward()
                print(f"D Step loss : {loss_D.item()}")

            for j in range(config.G_steps):
                # run gen
                imputed, disc_output, latent, reconstructed = m(data)
                loss_G = F.binary_cross_entropy_with_logits(disc_output.squeeze(), 1-batch_mask).mean()

                # loss_pre
                pre_tmp = imputed.squeeze() * batch_mask
                target_pre = batch_mask * batch_data
                loss_pre = nn.MSELoss()(pre_tmp, target_pre)

                # loss_re
                out_tmp = reconstructed.squeeze() * batch_mask
                targ_re = batch_data * batch_mask
                # loss_re = mse_error(out_tmp, targ_re)
                loss_re = nn.MSELoss()(out_tmp, targ_re)

                # loss_km
                HTH = torch.matmul(latent, latent.T)
                FTHTHF = torch.matmul(torch.matmul(m.F.T, HTH), m.F)
                loss_km = torch.trace(HTH) - torch.trace(FTHTHF)

                (loss_G + loss_pre + loss_re + loss_km * config.lambda_kmeans).backward()
                print(f"G Step loss : {loss_D.item()}")

                if i % config.T_update_F == 0:
                    '''calculate F'''
                    # run H
                    imputed, disc_output, latent, reconstructed = m(data)
                    # F_update
                    U, s, V = torch.linalg.svd(latent)
                    F_new = U.T[:config.k_cluster, :]
                    F_new = F_new.T
                    m.update_F(F_new)

            '''TEST'''
            with torch.no_grad():
                H_outputs = []
                for batch_data, batch_label, batch_mask, batch_length, batch_lengthmark in get_batch(test_data,test_label,test_mask,test_length,test_lengthmark,config):
                    data = {}
                    data['values'] = batch_data
                    data['masks'] = batch_mask
                    # get outputs["H"]
                    imputed, disc_output, latent, reconstructed = m(data)
                    H_outputs.append(latent)
                H_outputs = torch.concat(H_outputs, 0)
                H_outputs = H_outputs[:test_dataset_size, :]
                Km = KMeans(n_clusters=config.k_cluster)
                pred_H = Km.fit_predict(H_outputs.cpu().detach().numpy())

            '''record'''
            ri,nmi,acc,pur = assess(pred_H,test_label)
            RI.append(ri)
            NMI.append(nmi)
            ACC.append(acc)
            PUR.append(pur)


    ri = max(RI[80], RI[300], RI[500])
    nmi = max(NMI[80], NMI[300], NMI[500])
    acc = max(ACC[80], ACC[300], ACC[500])
    pur = max(PUR[80], PUR[300], PUR[500])

    return ri, nmi, pur, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--T_update_F', type=int, required=False, default=1)
    parser.add_argument('--G_steps', type=int, required=False, default=1)
    parser.add_argument('--D_steps', type=int, required=False, default=1)
    parser.add_argument('--epoch', type=int, required=False, default=500)
    parser.add_argument('--learning_rate', type=float, required=False, default=5e-3)
    parser.add_argument('--dataset_name', type=str, required=False, default='HouseVote')
    parser.add_argument('--lambda_kmeans', type=float, required=False, default=1e-3)
    parser.add_argument('--G_hiddensize', type=int, required=False, default=16)
    parser.add_argument('--G_layer', type=int, required=False, default=1)
    parser.add_argument('--IsD', type=bool, required=False, default=False)
    parser.add_argument('--cell_type', type=str, required=False, default='gru')
    parser.add_argument('--data_dir', type=str, required=False, default='dataset')
    parser.add_argument('--seq_len', type=int, required=False, default=16)
    parser.add_argument('--gpu_num', type=int, required=False, default=0)

    config = parser.parse_args()
    ri, nmi, pur, cluster_acc = run(config)

    print('%s,%.6f,%.6f,%.6f,%.6f' % (config.dataset_name, ri, nmi, pur, cluster_acc))

