# core framework follows https://github.com/tkipf/pygcn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
import msg


class GCN(nn.Module):
    def __init__(self, n_feats, e_feats, n_output, e_output,
                 h_size, n_hids, n_steps,
                 act=F.leaky_relu, layer_norm_on=False):
        '''
        n_feats: number of node features
        e_feats: number of edge features
        n_output: output size of node
        e_output: output size of edge
        h_size: feature size in hidden space
        n_hids: number of hidden neurons (a list)
        n_steps: number of message passing steps

        There are 7 transoformations: f, g, h, k1, k2, l, q
        1. y = f(x), raises the dimension of node features
        2. y <- g(sum_neighbors y) + y, message passing
        3. z = h(y), mapping to node output
        4. z_ij = q[k1(y_i) + k2(y_j) + l(x_ij)],
           mapping to edge output

        Note: if e_feats is 0, we won't have the term k(x_ij) in (4) above,
        so the number of transformation reduces to 6

        Note: for simplicity, we create the same neural network
        structure (with different set of parameters) for all
        transformation (specified by the list in n_hids).
        '''

        super(GCN, self).__init__()

        self.msg = msg.MessagePassing(
            n_feats, n_hids, h_size, n_steps, act, layer_norm_on)
        self.h = layers.Transformation(
            h_size, n_hids, n_output, act, layer_norm_on)
        self.k1 = layers.Transformation(
            h_size, n_hids, h_size, act, layer_norm_on)
        self.k2 = layers.Transformation(
            h_size, n_hids, h_size, act, layer_norm_on)
        self.l = layers.Transformation(
            e_feats, n_hids, h_size, act, layer_norm_on)
        self.q = layers.Transformation(
            h_size, n_hids, e_output, act, layer_norm_on)

    def forward(self, node_feats, adj_mat, edges, edge_feats):
        '''
        node_feats: node features, size n * d
        adj_mat: adjacancy matrix, size n * n
        edges: list of edges, each edge represented by node-tuple (i, j)
        edge_feats: edge features corresponding to the node-tuples
        '''
        num_nodes = node_feats.shape[0]

        # message passing among nodes
        node_embeddings = self.msg(node_feats, adj_mat)

        # map to node outputs
        node_outputs = self.h(node_embeddings)

        # edge embedding
        node_mat_1, node_mat_2 = convert_sp_node_mat(num_nodes, edges)
        left_node_embeddings = self.k1(torch.spmm(node_mat_1, node_embeddings))
        right_node_embeddings = self.k2(torch.spmm(node_mat_2, node_embeddings))
        edge_embeddings = left_node_embeddings + right_node_embeddings
        if edge_feats is not None:
            edge_embeddings = edge_embeddings + self.l(edge_feats)

        # map to edge outputs
        edge_outputs = self.q(edge_embeddings)

        return node_outputs, edge_outputs


def convert_sp_node_mat(num_nodes, list_edges):
    '''
    list_edges: list of node-tuples

    Example: list_edges = [(0, 2), (0, 1), (1, 2), (2, 0)]
             num_noes = 3
    
    Note that there are two edges of opposite directions
    in between node 0 and 2. 

    The outputs are two sparse matrices of size
    num_nodes * num_edges mapping nodes to end of edges.

    For this example, the outputs are

    [[1, 0, 0],          [[0, 0, 1],
     [1, 0, 0],    and    [0, 1, 0],
     [0, 1, 0],           [0, 0, 1],
     [0, 0, 1]]           [1, 0, 0]]
    '''
    l = len(list_edges)
    edge_ends = np.array(list_edges)
    node_indices = range(l)
    ones = torch.FloatTensor([1 for _ in range(l)])
    mat_size = torch.Size([l, num_nodes])

    node_mat_1 = torch.sparse.FloatTensor(
        torch.LongTensor([node_indices, edge_ends[:, 0]]),
        ones, mat_size)

    node_mat_2 = torch.sparse.FloatTensor(
        torch.LongTensor([node_indices, edge_ends[:, 1]]),
        ones, mat_size)

    return node_mat_1, node_mat_2
