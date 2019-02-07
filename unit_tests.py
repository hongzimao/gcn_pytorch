import unittest
import numpy as np
import torch
import layers
import msg
import gcn

# -------------------------------------------------------------------------

class TestTransformation(unittest.TestCase):

    def test_identity_transformation(self):
        f = layers.Transformation(3, [3], 3)
        eye_weights(f)
        in_vec = torch.FloatTensor([[1, 2, 3],
                                    [4, 5, 6]])
        out_vec = f(in_vec)
        self.assertTrue(bool(torch.all(
            torch.eq(in_vec, out_vec))))

    def test_known_transformation(self):
        f = layers.Transformation(2, [2], 2)
        f.weights[0].data = torch.FloatTensor([[1, 2], [-3, -4]])
        f.weights[1].data = torch.FloatTensor([[-5, 6], [-7, 8]])
        f.biases[0].data = torch.FloatTensor([-2, 1])
        f.biases[1].data = torch.FloatTensor([4, -3])
        in_vec = torch.FloatTensor([[9, 10]])
        out_vec = f(in_vec)
        self.assertTrue(bool(torch.all(torch.eq(out_vec,
            torch.FloatTensor([[4, 0]])))))

    def test_no_hidden_layer(self):
        f = layers.Transformation(2, [], 2)
        self.assertEqual(len(f.weights), 1)
        self.assertEqual(len(f.biases), 1)
        f.weights[0].data.zero_()
        f.weights[0].data[0][0] = -1
        f.weights[0].data[1][1] = -1
        in_vec = torch.FloatTensor([[1, 2]])
        out_vec = f(in_vec)
        self.assertTrue(bool(torch.all(torch.eq(out_vec,
            torch.FloatTensor([[0, 0]])))))

# -------------------------------------------------------------------------

class TestMessagePassing(unittest.TestCase):

    def test_identity_sum_features_one_step(self):
        msg_pass = msg.MessagePassing(2, [2], 2, 1)
        eye_weights(msg_pass.f)
        eye_weights(msg_pass.g)
        node_feats = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
        adj_mat = torch.FloatTensor([[0, 1, 1],
                                     [1, 0, 0],
                                     [1, 0, 0]])
        out_vec = msg_pass(node_feats, adj_mat)
        self.assertTrue(bool(torch.all(torch.eq(out_vec,
            torch.FloatTensor([[9, 12], [4, 6], [6, 8]])))))

    def test_identity_sum_features_two_steps(self):
        msg_pass = msg.MessagePassing(2, [2], 2, 2)
        eye_weights(msg_pass.f)
        eye_weights(msg_pass.g)
        node_feats = torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        adj_mat = torch.FloatTensor([[0, 1, 1, 1],
                                     [1, 0, 0, 0],
                                     [1, 0, 0, 1],
                                     [1, 0, 1, 0]])
        out_vec = msg_pass(node_feats, adj_mat)
        self.assertTrue(bool(torch.all(torch.eq(out_vec,
            torch.FloatTensor([[46, 58], [20, 26], [42, 52], [42, 52]])))))

# -------------------------------------------------------------------------

class TestGCN(unittest.TestCase):

    def test_identity_sum_features(self):
        graph_cnn = gcn.GCN(2, 2, 2, 2, 2, [2], 1)
        eye_weights(graph_cnn.msg.f)
        eye_weights(graph_cnn.msg.g)
        eye_weights(graph_cnn.h)
        eye_weights(graph_cnn.k1)
        eye_weights(graph_cnn.k2)
        eye_weights(graph_cnn.l)
        eye_weights(graph_cnn.q)
        node_feats = torch.FloatTensor([[1, 2], [-3, -4], [5, 6]])
        adj_mat = torch.FloatTensor([[0, 1, 1],
                                     [1, 0, 0],
                                     [1, 0, 0]])
        edges = [(0, 1), (0, 2), (1, 0), (2, 0)]
        edge_feats = torch.FloatTensor([[-1, 2], [3, -4], [-5, 6], [7, -8]])
        node_output, edge_output = graph_cnn(
            node_feats, adj_mat, edges, edge_feats)
        self.assertTrue(bool(torch.all(torch.eq(node_output,
            torch.FloatTensor([[6, 8], [1, 2], [6, 8]])))))
        self.assertTrue(bool(torch.all(torch.eq(edge_output,
            torch.FloatTensor([[7, 12], [15, 16], [7, 16], [19, 16]])))))


    def test_zero_output(self):
        graph_cnn = gcn.GCN(3, 0, 0, 0, 16, [16], 5)
        node_feats = torch.FloatTensor(4, 3)
        node_feats.data.uniform_(-1, 1)
        adj_mat = torch.ones([4, 4])
        edges = [(0, 2), (0, 1), (1, 2), (2, 0)]
        edge_feats = None
        node_output, edge_output = graph_cnn(
            node_feats, adj_mat, edges, edge_feats)
        self.assertEqual(node_output.shape[1], 0)
        self.assertEqual(edge_output.shape[1], 0)

# -------------------------------------------------------------------------

class TestTrainGCN(unittest.TestCase):

    def test_node_output(self):
        seed = 4
        n_hids = [16]
        n_iters = 500

        torch.manual_seed(seed)
        np.random.seed(seed)

        graph_cnn = gcn.GCN(2, 0, 1, 0, 2, n_hids, 1)
        graph_cnn.train()
        optimizer = torch.optim.Adam(graph_cnn.parameters(), lr=1e-2)

        losses = []
        for _ in range(n_iters):

            node_feats_np = np.random.rand(3, 2)
            adj_mat_np = np.array([[0, 1, 1],
                                   [1, 0, 0],
                                   [1, 0, 0]])
            edges = [(0, 1), (0, 2), (1, 0), (2, 0)]

            node_feats = torch.FloatTensor(node_feats_np)
            adj_mat = torch.FloatTensor(adj_mat_np)

            node_output, edge_output = graph_cnn(node_feats, adj_mat, edges, None)

            true_node_output = torch.FloatTensor(
                np.sum(adj_mat_np.dot(node_feats_np), axis=1, keepdims=True))

            loss = torch.sum((true_node_output - node_output) ** 2)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # < 5% error
        self.assertTrue(bool(losses[0] > 0.5))
        self.assertTrue(bool((sum(losses[-10:]) / 10) < 0.05))

    def test_edge_ouotput(self):
        seed = 4
        n_hids = [16]
        n_iters = 500

        torch.manual_seed(seed)
        np.random.seed(seed)

        graph_cnn = gcn.GCN(2, 2, 0, 1, 2, n_hids, 1)
        graph_cnn.train()
        optimizer = torch.optim.Adam(graph_cnn.parameters(), lr=1e-2)

        losses = []
        for _ in range(n_iters):

            node_feats_np = np.random.rand(3, 2)
            adj_mat_np = np.array([[0, 1, 1],
                                   [1, 0, 0],
                                   [1, 0, 0]])
            edges = [(0, 1), (0, 2), (1, 0), (2, 0)]

            node_feats = torch.FloatTensor(node_feats_np)
            adj_mat = torch.FloatTensor(adj_mat_np)

            node_output, edge_output = graph_cnn(node_feats, adj_mat, edges, None)

            # left and right node contributes differently
            true_edge_output_np = np.array([2 * node_feats_np[0, :] + node_feats_np[1, :],
                                            2 * node_feats_np[0, :] + node_feats_np[2, :],
                                            2 * node_feats_np[1, :] + node_feats_np[0, :],
                                            2 * node_feats_np[2, :] + node_feats_np[0, :]])

            true_edge_output_np = np.sum(true_edge_output_np, axis=1, keepdims=True)
            true_edge_output = torch.FloatTensor(true_edge_output_np)

            loss = torch.sum((true_edge_output - edge_output) ** 2)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # < 5% error
        self.assertTrue(bool(losses[0] > 0.5))
        self.assertTrue(bool((sum(losses[-10:]) / 10) < 0.05))

# -------------------------------------------------------------------------

def eye_weights(f):
    for i in range(len(f.weights)):
        assert f.weights[i].shape[0] == f.weights[i].shape[1]
        f.weights[i].data.zero_()
        for j in range(f.weights[i].shape[0]):
            f.weights[i].data[j][j] = 1

# -------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
