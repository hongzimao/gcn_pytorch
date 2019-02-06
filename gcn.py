# core framework follows https://github.com/tkipf/pygcn

import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, n_feats, e_feats, n_output, e_output, num_hids, n_steps):
        '''
        n_feats: number of node features
        e_feats: number of edge features
        n_output: output size of node
        e_output: output size of edge
        num_hids: number of hidden neurons (a list)
        n_steps: number of message passing steps

        There are 7 transoformations: f, g, h, k1, k2, k, l
        1. y = f(x), raises the dimension of node features
        2. y <- g(sum_neighbors y) + y, message passing
        3. z = h(y), mapping to node output
        4. z_ij = l[k1(y_i) + k2(y_j) + k(x_ij)],
           mapping to edge output

        Note: if e_feats is 0, we won't have the term k(x_ij) in (4) above,
        so the number of transformation reduces to 6

        Note: for simplicity, we create the same neural network
        structure (with different set of parameters) for all
        transformation (specified by the list in num_hids).
        '''

        super(GCN, self).__init__()
