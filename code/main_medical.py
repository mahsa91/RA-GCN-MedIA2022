import time
import argparse

import torch
seed_num = 17
torch.manual_seed(seed_num)
import torch.nn.functional as F

import itertools as it
import matplotlib.pyplot as plt

from code.model import RAGCN
from code.utils import accuracy, load_data_medical, encode_onehot_torch, class_f1, auc_score
from code.model import my_sigmoid


def train():
    ### create structure for discriminator and weighting networks
    struc_D = {'dropout': args.dropout_D, 'wd': 5e-4, 'lr': args.lr_D, 'nhid': structure_D}
    ### For simplicity, all weighting networks have the same hyper-parameters. They can be defined as a list of /
    # / dictionaries which each dictionary contains the hyper-parameters of each weighting network
    struc_Ws = n_ws*[{'dropout': args.dropout_W,  'wd': 5e-4, 'lr': args.lr_W, 'nhid': structure_W}]
    ### stats variable keeps the statistics of the network on train, validation and test sets
    stats = dict()
    ### act is the function for normalization of weights of samples in each class
    act = my_sigmoid
    ### Definition of model
    model = RAGCN(adj=adj, features=features, nclass=nclass, struc_D=struc_D, struc_Ws=struc_Ws,n_ws=n_ws,
                 weighing_output_dim=1, act=act, gamma=args.gamma)

    if use_cuda:
        model.cuda()

    ### Keeping the best stats based on the value of Macro F1 in validation set
    max_val = dict()
    max_val['f1Macro_val'] = 0
    for epoch in range(args.epochs):
        model.train()
        ### Train discriminator and weighting networks
        model.run_both(epoch_for_D=args.epoch_D, epoch_for_W=args.epoch_W, labels_one_hot=labels_one_hot[idx_train, :],
                       samples=idx_train, args_cuda=use_cuda, equal_weights=False)

        model.eval()
        ### calculate stats for training set
        class_prob, embed = model.run_D(samples=idx_train)
        weights, _ = model.run_W(samples=idx_train, labels=labels[idx_train], args_cuda=use_cuda, equal_weights=False)
        stats['loss_train'] = model.loss_function_D(class_prob, labels_one_hot[idx_train], weights).item()
        stats['nll_train'] = F.nll_loss(class_prob, labels[idx_train]).item()
        stats['acc_train'] = accuracy(class_prob, labels=labels[idx_train]).item()
        stats['f1Macro_train'] = class_f1(class_prob, labels[idx_train], type='macro')
        if nclass == 2:
            stats['f1Binary_train'] = class_f1(class_prob, labels[idx_train], type='binary', pos_label=pos_label)
            stats['AUC_train'] = auc_score(class_prob, labels[idx_train])

        ### calculate stats for validation and test set
        test(model, stats)
        ### Drop first epochs and keep the best based on the macro F1 on validation set just for reporting
        if epoch > drop_epochs and max_val['f1Macro_val'] < stats['f1Macro_val']:
            for key, val in stats.items():
                max_val[key] = val

        ### Print stats in each epoch
        print('Epoch: {:04d}'.format(epoch + 1))
        print('acc_train: {:.4f}'.format(stats['acc_train']))
        print('f1_macro_train: {:.4f}'.format(stats['f1Macro_train']))
        print('loss_train: {:.4f}'.format(stats['loss_train']))

    ### Reporting the best results on test set
    print('========Results==========')
    for key, val in max_val.items():
        if 'loss' in key or 'nll' in key or 'test' not in key:
            continue
        print(key.replace('_', ' ') + ' : ' + str(val))


### Calculate metrics on validation and test sets
def test(model, stats):
    model.eval()

    class_prob, embed = model.run_D(samples=idx_val)
    weights, _ = model.run_W(samples=idx_val, labels=labels[idx_val], args_cuda=use_cuda, equal_weights=True)

    stats['loss_val'] = model.loss_function_D(class_prob, labels_one_hot[idx_val], weights).item()
    stats['nll_val'] = F.nll_loss(class_prob, labels[idx_val]).item()
    stats['acc_val'] = accuracy(class_prob, labels[idx_val]).item()
    stats['f1Macro_val'] = class_f1(class_prob, labels[idx_val], type='macro')
    if nclass == 2:
        stats['f1Binary_val'] = class_f1(class_prob, labels[idx_val], type='binary', pos_label=pos_label)
        stats['AUC_val'] = auc_score(class_prob, labels[idx_val])

    class_prob, embed = model.run_D(samples=idx_test)
    weights, _ = model.run_W(samples=idx_test, labels=labels[idx_test], args_cuda=use_cuda, equal_weights=True)

    stats['loss_test'] = model.loss_function_D(class_prob, labels_one_hot[idx_test], weights).item()
    stats['nll_test'] = F.nll_loss(class_prob, labels[idx_test]).item()
    stats['acc_test'] = accuracy(class_prob, labels[idx_test]).item()
    stats['f1Macro_test'] = class_f1(class_prob, labels[idx_test], type='macro')
    if nclass == 2:
        stats['f1Binary_test'] = class_f1(class_prob, labels[idx_test], type='binary', pos_label=pos_label)
        stats['AUC_test'] = auc_score(class_prob, labels[idx_test])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--epoch_D', type=int, default=1,
                        help='Number of training loop for discriminator in each epoch.')
    parser.add_argument('--epoch_W', type=int, default=1,
                        help='Number of training loop for discriminator in each epoch.')
    parser.add_argument('--lr_D', type=float, default=0.01,
                        help='Learning rate for discriminator.')
    parser.add_argument('--lr_W', type=float, default=0.01,
                        help='Equal learning rate for weighting networks.')
    parser.add_argument('--dropout_D', type=float, default=0.5,
                        help='Dropout rate for discriminator.')
    parser.add_argument('--dropout_W', type=float, default=0.5,
                        help='Dropout rate for weighting networks.')
    parser.add_argument('--gamma', type=float, default=1,
                        help='Coefficient of entropy term in loss function.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    ### This list shows the number of hidden neurons in each hidden layer of discriminator
    structure_D = [2]
    ### This list shows the number of hidden neurons in each hidden layer of weighting networks
    structure_W = [4]
    ### The results of first drop_epochs will be dropped for choosing the best network based on the validation set
    drop_epochs = 500
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    ### Loading function should return the following variables
    ### adj is a dictionary including 'D' and 'W' as keys
    ### adj['D'] contains the main adjacency matrix between all samples for discriminator
    ### adj['W'] contains a list of adjacency matrices. Element i contains the adjacency matrix between samples of /
    ### / class i in the training samples
    ### Features is a tensor with size N by F
    ### labels is a list of node labels
    ### idx train is a list contains the index of training samples. idx_val and idx_test follow the same pattern
    adj, features, labels, idx_train, idx_val, idx_test = load_data_medical(dataset_addr='../data/synthetic/per-90gt-0.5.pkl',
                                                                            train_ratio=0.6, test_ratio=0.2)

    ### start of code
    labels_one_hot = encode_onehot_torch(labels)
    nclass = labels_one_hot.shape[1]
    n_ws = nclass
    pos_label = None
    if nclass == 2:
        pos_label = 1
        zero_class = (labels == 0).sum()
        one_class = (labels == 1).sum()
        if zero_class < one_class:
            pos_label = 0
    if use_cuda:
        for key, val in adj.items():
            if type(val) is list:
                for i in range(len(adj)):
                    adj[key][i] = adj[key][i].cuda()
            else:
                adj[key] = adj[key].cuda()
        features = features.cuda()
        labels_one_hot = labels_one_hot.cuda()
        labels = labels.cuda()

    ### Training the networks
    train()


