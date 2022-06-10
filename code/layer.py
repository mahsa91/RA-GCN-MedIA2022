import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, order, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        self.weight = torch.nn.ParameterList([])
        for i in range(self.order):
            self.weight.append(Parameter(torch.FloatTensor(in_features, out_features)))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.order):
            stdv = 1. / math.sqrt(self.weight[i].size(1))
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = []
        if self.order == 1 and type(adj) != list:
            adj = [adj]
        for i in range(self.order):
            support = torch.mm(input, self.weight[i])
            # output.append(support)
            output.append(torch.mm(adj[i], support))
        output = sum(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
