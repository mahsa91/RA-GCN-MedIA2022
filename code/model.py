import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from code.layer import GraphConvolution


def my_sigmoid(mx, dim):
    mx = torch.sigmoid(mx)
    return F.normalize(mx, p=1, dim=dim)


class Discriminator(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, order):
        super(Discriminator, self).__init__()

        layers = []
        if len(nhid) == 0:
            layers.append(GraphConvolution(nfeat, nclass, order=order))
        else:
            layers.append(GraphConvolution(nfeat, nhid[0], order=order))
            for i in range(len(nhid) - 1):
                layers.append(GraphConvolution(nhid[i], nhid[i + 1], order=order))
        if nclass > 1:
            layers.append(GraphConvolution(nhid[-1], nclass, order=order))
        self.gc = nn.ModuleList(layers)

        self.dropout = dropout
        self.nclass = nclass

    def forward(self, x, adj, samples=-1, func=F.relu):
        end_layer = len(self.gc) - 1 if self.nclass > 1 else len(self.gc)
        for i in range(end_layer):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc[i](x, adj)
            x = func(x)

        if self.nclass > 1:
            classifier = self.gc[-1](x, adj)
            classifier = F.log_softmax(classifier, dim=1)
            return classifier[samples,:], x
        else:
            return None, x


class Weighing(nn.Module):
    def __init__(self, nfeat, nhid, weighing_output_dim, dropout, order):
        super(Weighing, self).__init__()

        layers = []
        layers.append(GraphConvolution(nfeat, nhid[0], order=order))
        for i in range(len(nhid) - 1):
            layers.append(GraphConvolution(nhid[i], nhid[i + 1], order=order))

        layers.append(GraphConvolution(nhid[-1], weighing_output_dim, order=order))
        self.gc = nn.ModuleList(layers)

        self.dropout = dropout

    def forward(self, x, adj, func, samples=-1):
        end_layer = len(self.gc) - 1
        for i in range(end_layer):
            x = self.gc[i](x, adj)
            x = F.leaky_relu(x, inplace=False)
            x = F.dropout(x, self.dropout, training=self.training)

        if type(samples) is int:
            weights = func(self.gc[-1](x, adj), dim=0)
        else:
            weights = func(self.gc[-1](x, adj)[samples, :], dim=0)

        if len(weights.shape) == 1:
            weights = weights[None, :]

        return weights, x


class RAGCN:
    def __init__(self, features, adj , nclass, struc_D, struc_Ws, n_ws, weighing_output_dim, act, gamma=1):
        nfeat = features.shape[1]
        if type(adj['D']) is list:
            order_D = len(adj['D'])
        else:
            order_D = 1
        if type(adj['W'][0]) is list:
            order_W = len(adj['W'][0])
        else:
            order_W = 1
        self.n_ws = n_ws
        self.wod = weighing_output_dim
        self.net_D = Discriminator(nfeat, struc_D['nhid'], nclass, struc_D['dropout'], order=order_D)
        self.gamma = gamma
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=struc_D['lr'], weight_decay=struc_D['wd'])
        self.net_Ws = []
        self.opt_Ws = []
        for i in range(n_ws):
            self.net_Ws.append(Weighing(nfeat=nfeat, nhid=struc_Ws[i]['nhid'], weighing_output_dim=weighing_output_dim
                              , dropout=struc_Ws[i]['dropout'], order=order_W))
            self.opt_Ws.append(
                optim.Adam(self.net_Ws[-1].parameters(), lr=struc_Ws[i]['lr'], weight_decay=struc_Ws[i]['wd']))
        self.adj_D = adj['D']
        self.adj_W = adj['W']
        self.features = features
        self.act = act

    def run_D(self, samples):
        class_prob, embed = self.net_D(self.features, self.adj_D, samples)
        return class_prob, embed

    def run_W(self, samples, labels, args_cuda=False, equal_weights=False):
        batch_size = samples.shape[0]
        embed = None
        if equal_weights:
            max_label = int(labels.max().item() + 1)
            weight = torch.empty(batch_size)
            for i in range(max_label):
                labels_indices = (labels == i).nonzero().squeeze()
                if len(labels_indices.shape) == 0:
                    batch_size = 1
                else:
                    batch_size = len(labels_indices)
                if labels_indices is not None:
                    weight[labels_indices] = 1 / batch_size * torch.ones(batch_size)
            weight = weight / max_label
        else:
            max_label = int(labels.max().item() + 1)
            weight = torch.empty(batch_size)
            if args_cuda:
                weight = weight.cuda()
            for i in range(max_label):
                labels_indices = (labels == i).nonzero().squeeze()
                if labels_indices is not None:
                    sub_samples = samples[labels_indices]
                    weight_, embed = self.net_Ws[i](x=self.features[sub_samples,:], adj=self.adj_W[i], samples=-1, func=self.act)
                    weight[labels_indices] = weight_.squeeze() if self.wod == 1 else weight_[:,i]
            weight = weight / max_label

        if args_cuda:
            weight = weight.cuda()

        return weight, embed

    def loss_function_D(self, output, labels, weights):
        return torch.sum(- weights * (labels.float() * output).sum(1), -1)

    def loss_function_G(self, output, labels, weights):
        return torch.sum(- weights * (labels.float() * output).sum(1), -1) - self.gamma*torch.sum(weights*torch.log(weights+1e-20))

    def zero_grad_both(self):
        self.opt_D.zero_grad()
        for opt in self.opt_Ws:
            opt.zero_grad()

    def run_both(self, epoch_for_D, epoch_for_W, labels_one_hot, samples=-1,
                 args_cuda=False, equal_weights=False):
        labels_not_onehot = labels_one_hot.max(1)[1].type_as(labels_one_hot)
        for e_D in range(epoch_for_D):
            self.zero_grad_both()

            class_prob_1, embed = self.run_D(samples)
            weights_1, _ = self.run_W(samples=samples, labels=labels_not_onehot, args_cuda=args_cuda, equal_weights=equal_weights)

            loss_D = self.loss_function_D(output=class_prob_1, labels=labels_one_hot, weights=weights_1)
            loss_D.backward()
            self.opt_D.step()

        for e_W in range(epoch_for_W):
            self.zero_grad_both()

            class_prob_2, embed = self.run_D(samples)
            weights_2, embeds = self.run_W(samples=samples, labels=labels_not_onehot, args_cuda=args_cuda, equal_weights=equal_weights)

            loss_G = -self.loss_function_G(output=class_prob_2, labels=labels_one_hot, weights=weights_2)
            loss_G.backward()
            for opt in self.opt_Ws:
                opt.step()

    def train(self):
        self.net_D.train()
        for i in range(self.n_ws):
            self.net_Ws[i].train()

    def eval(self):
        self.net_D.eval()
        for i in range(self.n_ws):
            self.net_Ws[i].eval()

    def cuda(self):
        self.features = self.features.to(device='cuda')
        for i in range(len(self.adj_D)):
            self.adj_D[i] = self.adj_D[i].to(device='cuda')
        for i in range(len(self.adj_W)):
            self.adj_W[i] = self.adj_W[i].to(device='cuda')
        self.net_D.cuda()
        for i in range(self.n_ws):
            self.net_Ws[i].cuda()