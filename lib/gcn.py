import scipy.sparse
import modules.graph as graph
from lib.nn import NN
import numpy as np
import tensorflow as tf

# TODO: test this module

class gcn_specs:
    """
    Use this data structure to describe how you wish a specific graph convolutional network to be built.
    """

    def __init__(self):

        # number of graph convolution layers:
        self.n_gconv_layers = 2

        # for an N-layer GCN, the four following variables need to be N-element lists.
        # a list of graph Laplacian matrices for each graph convolution layer:
        self.laplacians = [ None, None ]
        # a list of numbers of graph convolution filters to use for each graph convolution  layer:
        self.n_gconv_filters = [64, 32]
        # a list of numbers of graph fourier transform polynomial orders to use for each graph convolution  layer:
        self.polynomial_orders = [3, 3]
        # a list of numbers of pooling sizes to use after each graph convolution layer:
        self.pooling_sizes = [1, 1]

        # number of fully connected layer dimsionalities
        self.fc_dims = [32, 16, 4]

        # specify which type of bias to use
        # valid values: 'per_filter', 'per_node_per_filter'
        self.bias_type = 'per_filter'

        # specify which type of pooling to use
        self.pool_fn = tf.nn.max_pool

        # specify activation function:
        self.activation_fn = tf.nn.relu

        # specify whether or not to apply batch normalization
        self.batch_norm = False

        # specify whether or not to apply L2 regularization
        self.regularize = False

class GCN(NN):
    """
    This class defines layers for constructing graph convolutional network

    conventions:
        input of a graph convolution layer is usually a NxMxF tensor
        N: number of samples (texts, images, etc.)
        M: number of graph nodes
        F: number of feature dimensions on each node
    """

    def __init__(self, specs: gcn_specs, is_training):
        NN.__init__(self)
        self.specs = specs
        self.is_training = is_training

    def create_placeholder(self, n_nodes, n_features, name):
        """create input placeholders"""
        ph = tf.placeholder([None, n_nodes, n_features], name)
        return ph

    def build(self, x, dropout):
        """
        build GCN according specifications
        :param x: input tensor or placeholder
        :param dropout: placeholder for dropout parameter
        :return: the desired graph network output
        """

        specs = self.specs
        regularizers = []

        # graph convolution layers
        for i in range(specs.n_gconv_layers):
            with tf.variable_scope('gcn_gconv{}'.format(i + 1)):
                with tf.name_scope('gconv'):
                    x, regs = self.gconv(x,
                                   specs.laplacians[i],
                                   specs.n_gconv_filters[i],
                                   specs.polynomial_orders[i],
                                   specs.regularize)
                    regularizers += regs
                with tf.name_scope('bias_activation'):
                    x, regs = self.bias(x,
                                  specs.bias_type,
                                  specs.activation_fn,
                                  specs.regularize)
                    regularizers += regs
                if specs.batch_norm: x = tf.layers.batch_normalization(x)
                with tf.name_scope('pooling'):
                    x = self.gpool(x,
                                   specs.pooling_sizes[i],
                                   specs.pool_fn)

        # fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M * F)])  # N x M
        for i, M in enumerate(specs.fc_dims):
            with tf.variable_scope('gcn_fc{}'.format(i + 1)):
                x, regs = self.fc(x, M, specs.activation_fn, specs.regularize)
                regularizers += regs
                if self.is_training: x = tf.nn.dropout(x, dropout)

        return x, regularizers

    # layer definitions

    def gconv(self, x, L, F_out, K, regularize=False):
        """
        The graph convolution layer.

        :param x: input tensor
        :param L: Laplacian matrix of the graph
        :param F_out: number of output filters of this layer
        :param K: polynomial order for for K-localisation
        :param regularize: whether or not to apply l2 regularizaiton
        :return: output tensor
        """

        N, M, F_in = x.get_shape()
        N, M, F_in = int(N), int(M), int(F_in)

        # Rescale the Laplacian matrix,
        L = scipy.sparse.csr_matrix(L) # make a copy
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # store as a TF sparse tensor.
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        # Fourier transformation on x to get its representation on Chebyshev basis
        # x_t is the transformed x
        x_0 = tf.transpose(x, [1, 2, 0])  # M x F_in x N
        x_0 = tf.reshape(x_0, [M, F_in * N])  # M x (F_in x N)
        x_t = tf.expand_dims(x_0, 0)

        if K > 1:
            x_1 = tf.sparse_tensor_dense_matmul(L, x_0)
            x_1_expanded = tf.expand_dims(x_1, 0)
            x_t = tf.concat([x_t, x_1_expanded], 0)
            for k in range(2, K):
                x_k = 2 * tf.sparse_tensor_dense_matmul(L, x_1) - x_0  # M x F_in*N
                x_k_expanded = tf.expand_dims(x_k, 0)
                x_t = tf.concat([x_t, x_k_expanded], 0)

        x_t = tf.reshape(x_t, [K, M, F_in, N])  # K x M x F_in x N
        x_t = tf.transpose(x_t, perm=[3, 1, 2, 0])  # N x M x F_in x K
        x_t = tf.reshape(x_t, [N * M, F_in * K])  # (N x M) x (F_in x K)

        # graph convolution
        W, regularizers = self.weight_variable([F_in * K, F_out], regularize=regularize)
        x_out = tf.matmul(x_t, W)  # (N x M) x F_out

        return tf.reshape(x_out, [N, M, F_out]), regularizers  # N x M x F_out

    def bias(self, x, bias_type, activation_fn, regularize=False):
        N, M, F = x.get_shape()
        if bias_type == 'per_filter':
            b, regularizers = self.bias_variable([1, 1, int(F)], regularize=regularize)
        elif bias_type == 'per_node_per_filter':
            b, regularizers = self.bias_variable([1, int(M), int(F)], regularize=regularize)
        else:
            raise Exception("Invalid bias type")
        x = x + b
        if activation_fn is not None:
            return activation_fn(x), regularizers
        else:
            return x, regularizers

    def gpool(self, x, p, pool_fn):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = pool_fn(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
            # tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

