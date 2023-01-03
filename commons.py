import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--T_update_F', type=int, required=False, default=1)
    parser.add_argument('--G_steps', type=int, required=False, default=1)
    parser.add_argument('--D_steps', type=int, required=False, default=1)
    parser.add_argument('--epoch', type=int, required=False, default=300)
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
    parser.add_argument('--nm_datasets', type=str, nargs='+', required=True, choices=[
        "Physionet 2012",
        "alizadeh-2000-v1-filter",
        "alizadeh-2000-v2-filter",
        "alizadeh-2000-v3-filter",
        "chen-2002-filter",
        "liang-2005-filter",
        "BloodSample",
        "HouseVote"]
    )

    # information of datasets in this paper
    dataset_info = {}
    dataset_info["Physionet 2012"] = {"input_dim": 35, "seq_len": 48}  # 0.92
    dataset_info["alizadeh-2000-v1-filter"] = {"input_dim": 1, "seq_len": 932}  # 0.94
    dataset_info["alizadeh-2000-v2-filter"] = {"input_dim": 1, "seq_len": 1030}  # 0.93
    dataset_info["alizadeh-2000-v3-filter"] = {"input_dim": 1, "seq_len": 1030}  # 0.92
    dataset_info["chen-2002-filter"] = {"input_dim": 1, "seq_len": 2328}  # 0.73
    dataset_info["liang-2005-filter"] = {"input_dim": 1, "seq_len": 2505}  # 0.95
    dataset_info["BloodSample"] = {"input_dim": 10, "seq_len": 20}  # 0.92
    dataset_info["HouseVote"] = {"input_dim": 1, "seq_len": 16}  # 0.95

    args = parser.parse_args()
    datasets = {}
    for nm in args.nm_datasets:
        datasets[nm] = dataset_info[nm]
    args.datasets
    return args


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
    mask_label = np.loadtxt(filename, delimiter = ",")
    mask = mask_label[:,1:].astype(np.float32)
    return mask


def load_data(filename):
    data_label = np.loadtxt(filename, delimiter = ",")
    data = data_label[:,1:].astype(np.float32)
    label = data_label[:,0].astype(np.int32)
    return data,label


def load_length(filename):
    length = np.loadtxt(filename, delimiter = ",")
    return length


def load_lengthmark(filename):
    lengthmark = np.loadtxt(filename,delimiter = ",")
    return lengthmark


def get_batch(data, mask, config):
    samples_num = data.shape[0]
    batch_num = int(samples_num / config.batch_size)
    left_row = samples_num - batch_num * config.batch_size
    device = torch.device(f'cuda:{config.gpu_num}' if torch.cuda.is_available() else 'cpu')

    data = torch.from_numpy(data).to(device)
    mask = torch.Tensor(mask).to(device)

    for i in range(batch_num):
        batch_data = data[i * config.batch_size: (i + 1) * config.batch_size, :]
        batch_mask = mask[i * config.batch_size: (i + 1) * config.batch_size, :]
        yield (batch_data, batch_mask)

    if left_row != 0:
        need_more = config.batch_size - left_row
        need_more = np.random.choice(np.arange(samples_num), size=need_more)
        batch_data = torch.concat((data[-left_row:, :], data[need_more,:]), axis=0)
        batch_mask = torch.concat((mask[-left_row:, :], mask[need_more,:]), axis=0)
        yield (batch_data, batch_mask)