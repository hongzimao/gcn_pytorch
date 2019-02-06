import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Transformation(nn.Module):
    def __init__(self, in_feats, num_hids, out_feats):
        '''
        in_feats: number of input features
        num_hids: number of hidden neurons (a list)
        out_feats: number of output features
        '''
        super(Transformation, self).__init__()

        self.in_feats = in_feats
        self.num_hids = num_hids
        self.out_feats = out_feats

        # parameter dimensions
        layers = [self.in_feats]
        layers.extend(self.num_hids)
        layers.append(self.out_feats)

        self.weights = []
        self.biases = []
        for l in range(len(layers) - 1):
            self.weights.append(Parameter(torch.FloatTensor(
                layers[l], layers[l + 1])))
            self.biases.append(Parameter(torch.FloatTensor(
                layers[l + 1])))

        # relu activation function
        self.act = F.relu

        # initialize parameters
        self.reset()

    def reset(self):
        # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
        for weight in self.weights:
            init_range = np.sqrt(6 / (weight.shape[0] + weight.shape[1]))
            weight.data.uniform_(-init_range, init_range)
        for bias in self.biases:
            bias.data.zero_()

    def forward(self, in_vec):
        x = in_vec
        for l in range(len(self.weights)):
            x = torch.mm(x, self.weights[l])
            x += self.biases[l]
            x = self.act(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_feats) + ' -> ' \
               + ' -> '.join(str(i) for i in self.num_hids) \
               + ' -> ' + str(self.out_feats) + ')'
