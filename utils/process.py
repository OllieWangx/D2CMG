import argparse
import torch
import numpy as np
import torch.nn as nn
import copy
import scipy.io as sio
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

BASEPATH = 'utils/data/'  # replace your path of datasets

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_A_r(adj, r):
    adj_label = adj
    for i in range(r - 1):
        adj_label = adj_label @ adj
    return adj_label


def local_preserve(x_dis, adj_label, tau=1.0):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss


def get_feature_dis(x, eps=1e-8):
    """
    x :           batch_size x nhid
    eps: small value to avoid division by 0 default 1e-8
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.clamp(x_sum, min=eps)  # clamp to avoid division by zero
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


def cluster(data, k, num_iter, cluster_temp, init):
    data = torch.diag(1. / (torch.norm(data, p=2, dim=1) + 1e-8)) @ data
    mu = init
    for t in range(num_iter):
        dist = data @ mu.t()
        r = torch.softmax(cluster_temp * dist, dim=1)
        cluster_r = r.sum(dim=0)
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        new_mu = torch.diag(1 / cluster_r) @ cluster_mean
        mu = new_mu
    dist = data @ mu.t()
    r = torch.softmax(cluster_temp * dist, 1)
    return mu, r, dist


def getParams(dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', default=dataset)
    parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
    parser.add_argument('--seed', type=int, default=2023, help='the seed to use')
    parser.add_argument('--test_epo', type=int, default=100, help='test_epo')
    parser.add_argument('--test_lr', type=int, default=0.01, help='test_lr')
    parser.add_argument('--sc', type=int, default=0, help='self connection')
    parser.add_argument('--eps', type=float, default=0.05, help='decoding error in rate distortion')
    if dataset == 'acm':
        parser.add_argument('--nb_epochs', type=int, default=400, help='the number of epochs')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--gama', type=float, default=0.05, help='compression of the node representations')
        parser.add_argument('--cluster_temp', type=float, default=10, help='cluster temp')
        parser.add_argument('--cfg', type=int, default=[256, 256, 128, 128], help='hidden dimension')
        parser.add_argument('--dropout', type=float, default=0.5, help='')
        parser.add_argument('--num_community', type=int, default=4, help='')
        parser.add_argument('--tau', type=float, default=1.0, help='')
        parser.add_argument('--A_r', type=int, default=3, help='')
        parser.add_argument('--weight_rd', type=float, default=0.0001, help='weight of loss for rd')
        parser.add_argument('--weight_ag', type=float, default=0.5, help='weight of loss for align')
    elif dataset == 'imdb4780':
        parser.add_argument('--nb_epochs', type=int, default=500, help='the number of epochs')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--gama', type=float, default=0.7, help='compression of the node representations')
        parser.add_argument('--cluster_temp', type=float, default=10, help='cluster temp')
        parser.add_argument('--cfg', type=int, default=[256, 256, 128, 128], help='hidden dimension')
        parser.add_argument('--dropout', type=float, default=0.5, help='')
        parser.add_argument('--num_community', type=int, default=4, help='')
        parser.add_argument('--tau', type=float, default=1.0, help='')
        parser.add_argument('--A_r', type=int, default=3, help='')
        parser.add_argument('--weight_rd', type=float, default=1e-3, help='weight of loss for rd')
        parser.add_argument('--weight_ag', type=float, default=1e-4, help='weight of loss for align')
    elif dataset == 'dblp4057':
        parser.add_argument('--nb_epochs', type=int, default=3500, help='the number of epochs')
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
        parser.add_argument('--gama', type=float, default=0.2, help='compression of the node representations')
        parser.add_argument('--cluster_temp', type=float, default=50, help='cluster temp')
        parser.add_argument('--cfg', type=int, default=[512, 512, 256, 256, 128, 128], help='hidden dimension')
        parser.add_argument('--dropout', type=float, default=0.1, help='')
        parser.add_argument('--num_community', type=int, default=32, help='')
        parser.add_argument('--tau', type=float, default=1.0, help='')
        parser.add_argument('--A_r', type=int, default=1, help='')
        parser.add_argument('--weight_rd', type=float, default=1e-5, help='weight of loss for rd')
        parser.add_argument('--weight_ag', type=float, default=1e-5, help='weight of loss for align')
    elif dataset == 'freebase':
        parser.add_argument('--nb_epochs', type=int, default=500, help='the number of epochs')
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
        parser.add_argument('--gama', type=float, default=0.7, help='compression of the node representations')
        parser.add_argument('--cluster_temp', type=float, default=10, help='cluster temp')
        parser.add_argument('--cfg', type=int, default=[256, 256, 256], help='hidden dimension')
        parser.add_argument('--dropout', type=float, default=0.1, help='')
        parser.add_argument('--num_community', type=int, default=4, help='')
        parser.add_argument('--tau', type=float, default=5.0, help='')
        parser.add_argument('--A_r', type=int, default=4, help='')
        parser.add_argument('--weight_rd', type=float, default=1e-2, help='weight of loss for rd')
        parser.add_argument('--weight_ag', type=float, default=1e-2, help='weight of loss for align')
    args, _ = parser.parse_known_args()
    return args


def load_acm_mat(sc=3):
    data = sio.loadmat(BASEPATH + 'acm.mat')
    label = data['label']

    adj_edge1 = data["PLP"]
    adj_edge2 = data["PAP"]
    adj_fusion1 = adj_edge1 + adj_edge2
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1

    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0]) * sc
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0]) * sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * sc

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_dblp4057(sc=3):
    data = sio.loadmat(BASEPATH + 'dblp4057.mat')
    label = data['label']

    adj_edge1 = data["net_APCPA"]
    adj_edge2 = data["net_APA"]
    adj_edge3 = data["net_APTPA"]
    adj_fusion1 = adj_edge1 + adj_edge2 + adj_edge3
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 3] = 0
    adj_fusion[adj_fusion == 3] = 1

    adj1 = adj_edge1 + np.eye(adj_edge1.shape[0]) * sc
    adj2 = adj_edge2 + np.eye(adj_edge2.shape[0]) * sc
    adj3 = adj_edge3 + np.eye(adj_edge3.shape[0]) * sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * sc

    truefeatures = data['features'].astype(float)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)
    adj_list = [adj1, adj2, adj3]
    adj_fusion = sp.csr_matrix(adj_fusion)
    truefeatures = sp.lil_matrix(truefeatures)

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_imdb4780(sc=3):
    data = sio.loadmat(BASEPATH + 'imdb4780.mat')
    label = data['label']
    ###########################################################
    adj_edge1 = data["MDM"]
    adj_edge2 = data["MAM"]
    adj_fusion1 = adj_edge1 + adj_edge2
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1
    ############################################################
    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0]) * sc
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0]) * sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * sc

    truefeatures = data['feature'].astype(float)

    truefeatures = sp.lil_matrix(truefeatures)

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)
    adj_list = [adj1, adj2]

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_freebase(sc=3):
    type_num = 3492
    ratio = [20, 40, 60]
    # The order of node types: 0 m 1 d 2 a 3 w
    path = BASEPATH + "freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_m = sp.eye(type_num)

    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")

    adj_fusion1 = np.array(mam.todense(), dtype=int) + np.array(mdm.todense(), dtype=int) + np.array(mwm.todense(),
                                                                                                     dtype=int)
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 3] = 0
    adj_fusion[adj_fusion == 3] = 1
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * sc

    adj1 = sp.csr_matrix(mam)
    adj2 = sp.csr_matrix(mdm)
    adj3 = sp.csr_matrix(mwm)
    adj_fusion = sp.csr_matrix(adj_fusion)

    # pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.FloatTensor(label)
    adj_list = [adj1, adj2, adj3]

    # pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]
    return adj_list, feat_m, label, train[0], val[0], test[0], adj_fusion


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    try:
        return features.todense()
    except:
        return features


def row_normalize(A):
    """Row-normalize dense matrix"""
    eps = 2.2204e-16
    rowsum = A.sum(dim=-1).clamp(min=0.) + eps
    r_inv = rowsum.pow(-1)
    A = r_inv.unsqueeze(-1) * A
    return A


def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
