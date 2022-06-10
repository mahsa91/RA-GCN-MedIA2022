import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import pickle


def encode_onehot_torch(labels):
    num_classes = int(labels.max() + 1)
    y = torch.eye(num_classes)
    return y[labels]


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def fill_features(features):
    empty_indices = np.argwhere(features != features)
    for index in empty_indices:
        features[index[0], index[1]] = np.nanmean(features[:,index[1]])

    return features


def load_data_medical(dataset_addr, train_ratio, test_ratio=0.2):
    with open(dataset_addr, 'rb') as f:  # Python 3: open(..., 'rb')
        adj, features, labels = pickle.load(f)
    n_node = adj.shape[1]
    adj = nx.adjacency_matrix(nx.from_numpy_array(adj))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj_D = normalize(adj + sp.eye(adj.shape[0]))
    adj_W = normalize(adj + sp.eye(adj.shape[0]))

    adj= dict()
    adj['D'] = adj_D

    features = features.astype(np.float)
    features = fill_features(features)

    idx_train, idx_test = train_test_split(range(n_node), test_size=test_ratio, random_state=42, stratify=labels)
    idx_train, idx_val = train_test_split(idx_train, train_size=train_ratio/(1-test_ratio), random_state=42, stratify=labels[idx_train])

    adj['W'] = []
    for nc in range(labels.max() + 1):
        nc_idx = np.where(labels[idx_train] == nc)[0]
        nc_idx = np.array(idx_train)[nc_idx]
        adj['W'].append(adj_W[np.ix_(nc_idx,nc_idx)])

    features = torch.FloatTensor(features)
    for key,val in adj.items():
        if key == 'D':
            adj[key] = torch.FloatTensor(np.array(adj[key].todense()))#sparse_mx_to_torch_sparse_tensor(adj[i])
        else:
            for i in range(len(val)):
                adj[key][i] = torch.FloatTensor(np.array(adj[key][i].todense()))

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def class_f1(output, labels, type='micro', pos_label=None):
    preds = output.max(1)[1].type_as(labels)
    if pos_label is None:
        return f1_score(labels.cpu().numpy(), preds.cpu(), average=type)
    else:
        return f1_score(labels.cpu().numpy(), preds.cpu(), average=type, pos_label=pos_label)


def auc_score(output, labels):
    return roc_auc_score(labels.cpu().numpy(), output[:,1].detach().cpu().numpy())


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)