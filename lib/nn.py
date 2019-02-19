import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

class NN:
    """Defines some useful functions for constructing neural networks"""

    __metaclass__ = ABCMeta  # this class is an abstract class

    def __init__(self):
        pass

    def weight_variable(self, shape, regularize, name='weights'):
        initializer = tf.contrib.layers.xavier_initializer()
        regularizers = []

        var = tf.get_variable(name, shape, tf.float32, initializer=initializer)
        if regularize:
            regularizers.append(tf.nn.l2_loss(var))
            # tf.summary.histogram(var.op.name, var)
        return var, regularizers

    def bias_variable(self, shape, regularize, name='bias'):
        initial = tf.constant_initializer(0.0)
        regularizers = []

        var = tf.get_variable(name, shape, tf.float32, initializer=initial)
        if regularize:
            regularizers.append(tf.nn.l2_loss(var))
            # tf.summary.histogram(var.op.name, var)
        return var, regularizers

    def fc(self, x, M_out, activation_fn, regularize=False, batch_norm=False):
        """Fully connected layer with M_out features."""
        N, M_in = x.get_shape()
        W, W_regs = self.weight_variable([int(M_in), M_out], regularize=regularize)
        b, b_regs = self.bias_variable([M_out], regularize=regularize)
        x = tf.matmul(x, W) + b
        regularizers = W_regs + b_regs
        if batch_norm: x = tf.layers.batch_normalization(x)
        if activation_fn is None:
            return x, regularizers
        else:
            return activation_fn(x), regularizers