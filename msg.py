import torch
import torch.nn as nn
import torch.nn.functional as F
import layers


class MessagePassing(nn.Module):
    def __init__(self, in_feats, n_hids, out_feats, num_steps,
                 act=F.leaky_relu, layer_norm_on=False):
        '''
        in_feats: number of input features
        n_hids: number of hidden neurons (a list)
        out_feats: number of output features

        f maps *all* raw feature vectors to size out_feats
        g performs message passing transformations

        y_i = f(x_i), for all i
        y_i <- g(sum_neighbors_j y_j) + y_i, at every step
        '''
        super(MessagePassing, self).__init__()

        self.num_steps = num_steps

        self.f = layers.Transformation(
            in_feats, n_hids, out_feats, act, layer_norm_on)
        self.g = layers.Transformation(
            out_feats, n_hids, out_feats, act, layer_norm_on)

    def forward(self, in_vec, adj_mat):
        x = self.f(in_vec)  # map features to hidden space
        for s in range(self.num_steps):
            ne = torch.mm(adj_mat, x)  # message passing
            ne = self.g(ne)
            x = x + ne

        return x
