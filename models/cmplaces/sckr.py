from lib.mlnet import MLNet
from modules.vgg19 import Vgg19
from modules.vgg16_trainable import Vgg16
from lib.gcn import gcn_specs, GCN
from scipy import sparse
import modules.graph as graph
import tensorflow as tf
import numpy as np

class Model(MLNet):
    """
    A baseline model for experimentation on eng-wiki

    modal 1:
        input: extracted text tf-idf features
        method: tf_idf
        output: tf_idf features

    modal 2:
        input: VGG19 extracted 4096 features
        method: VGG19
        output: vgg19 features

    conventions:
        input of a graph convolution layer is usually a NxMxF tensor
        N: number of samples (texts, images, etc.)
        M: number of graph nodes in each sample
        F: number of feature dimensions on each node
    """

    def __init__(self, adjmat_path, batch_size, desc_dims, out_dims, is_training=False, is_retrieving=False):
        MLNet.__init__(self, batch_size, desc_dims, out_dims, is_training, is_retrieving)
        self.adjmat_path = adjmat_path

    def build_modal_1(self):
        M_text = 4412
        F_text = 50
        n_gconv_layers = 2

        ph_text = tf.placeholder(tf.float32, [self.batch_size, M_text, F_text], 'input_text')

        # laplacians
        A = graph.load_adjmat(self.adjmat_path, M_text)
        A_sparse = sparse.csr_matrix(A).astype(np.float32)
        As = [A_sparse] * n_gconv_layers
        laplacians = [graph.laplacian(A, normalized=True) for A in As]

        # specify GCN parameters
        specs = gcn_specs()
        specs.n_gconv_layers = n_gconv_layers
        specs.laplacians = laplacians
        specs.n_gconv_filters = [1, 1]
        specs.polynomial_orders = [3, 3]
        specs.pooling_sizes = [1, 1]
        specs.fc_dims = []
        specs.bias_type = 'per_node_per_filter'
        specs.pool_fn = tf.nn.max_pool
        specs.activation_fn = tf.nn.relu
        specs.batch_norm = False
        specs.regularize = False

        # build GCN
        self.gcn = GCN(specs, is_training=self.is_training)
        gcn_out, regularizers = self.gcn.build(ph_text, self.ph_dropout)
        self.regularizers += regularizers

        return [ph_text], gcn_out

    def build_modal_2(self):
        F_image = 4096

        ph_image = tf.placeholder(tf.float32, [self.batch_size, F_image], 'input_image')

        return [ph_image], tf.squeeze(ph_image)
