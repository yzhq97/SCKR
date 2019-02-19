# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import time
from lib.nn import NN
from lib.utils import get_available_cpus
from lib.utils import get_available_gpus
from abc import ABCMeta, abstractmethod
from sklearn.metrics import roc_curve, auc
import os
import math

class MLNet(NN):
    """
    Metric Learning Net
    This class defines the basic structure of a dual-input metric learning network.
    The two feature extraction modals needs implementation.
    The descriptors produced by the two feature extraction modals should have exact same dimensions.
    """

    __metaclass__ = ABCMeta # this class is an abstract class

    def __init__(self, batch_size, desc_dims, out_dims, is_training=False, is_retrieving=False):
        NN.__init__(self)

        self.regularizers = []
        self.batch_size = batch_size
        self.desc_dims = desc_dims
        self.out_dims = out_dims
        self.is_training = is_training
        self.is_retrieving = is_retrieving

        tf.set_random_seed(int(1e6 * (time.time() % 1)))

        if is_training and is_retrieving:
            raise Exception("can not retrieve while training")

    @abstractmethod
    def build_modal_1(self): return [], None

    @abstractmethod
    def build_modal_2(self): return [], None

    def build(self, modal_1_device=None, modal_2_device=None, metrics_device=None):

        self.ph_dropout = tf.placeholder(tf.float32, [], 'dropout')
        self.ph_labels = tf.placeholder(tf.int32, [self.batch_size], 'labels')

        if self.is_retrieving:
            self.ph_desc_1 = tf.placeholder(tf.float32, [self.batch_size, self.desc_dims], 'desc_1')
            self.ph_desc_2 = tf.placeholder(tf.float32, [self.batch_size, self.desc_dims], 'desc_2')

        with tf.variable_scope('modal_1'), tf.device(modal_1_device):
            self.ph1, self.modal_1 = self.build_modal_1()

        with tf.variable_scope('fc_1'):
            self.descriptors_1, regularizers = self.fc(self.modal_1, self.desc_dims, activation_fn=None)
            self.regularizers += regularizers

        with tf.variable_scope('modal_2'), tf.device(modal_2_device):
            self.ph2, self.modal_2 = self.build_modal_2()

        with tf.variable_scope('fc_2'):
            self.descriptors_2, regularizers = self.fc(self.modal_2, self.desc_dims, activation_fn=None)
            self.regularizers += regularizers

        desc_1 = self.ph_desc_1 if self.is_retrieving else self.descriptors_1
        desc_2 = self.ph_desc_2 if self.is_retrieving else self.descriptors_2

        with tf.device(metrics_device):
            with tf.variable_scope('metrics'):
                x = tf.multiply(desc_1, desc_2)
                if self.is_training:
                    x = tf.nn.dropout(x, self.ph_dropout)
                with tf.variable_scope('fc'):
                    x, regularizers = self.fc(x, self.out_dims, activation_fn=None)
                    self.regularizers += regularizers
                self.logits = tf.squeeze(x)

        self.build_saver()

    def build_loss(self, lamda, mu, reg_weight):
        """Adds to the inference model the layers required to generate loss."""

        with tf.name_scope('loss'):
            with tf.name_scope('var_loss'):
                labels = tf.cast(self.ph_labels, tf.float32)
                shape = labels.get_shape()

                same_class = tf.boolean_mask(self.logits, tf.equal(labels, tf.ones(shape)))
                diff_class = tf.boolean_mask(self.logits, tf.not_equal(labels, tf.ones(shape)))
                same_mean, same_var = tf.nn.moments(same_class, [0])
                diff_mean, diff_var = tf.nn.moments(diff_class, [0])
                var_loss = same_var + diff_var

            with tf.name_scope('mean_loss'):
                mean_loss = lamda * tf.where(
                    tf.greater(mu - (same_mean - diff_mean), 0),
                    mu - (same_mean - diff_mean), 0)

            self.loss = (1) * var_loss + (1) * mean_loss
            regularize, regularization = len(self.regularizers) > 0, None
            if regularize:
                with tf.name_scope('regularization'):
                    regularization = reg_weight * tf.add_n(self.regularizers)
                self.loss += regularization

            # Summaries for TensorBoard.
            tf.summary.scalar('total', self.loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                if regularize:
                    operations = [var_loss, mean_loss, regularization, self.loss]
                else:
                    operations = [var_loss, mean_loss, self.loss]
                op_averages = averages.apply(operations)
                tf.summary.scalar('var_loss', averages.average(var_loss))
                tf.summary.scalar('mean_loss', averages.average(mean_loss))
                if regularize:
                    tf.summary.scalar('regularization', averages.average(regularization))
                tf.summary.scalar('total', averages.average(self.loss))
                with tf.control_dependencies([op_averages]):
                    self.loss_average = tf.identity(averages.average(self.loss), name='control')

    def build_train(self, learning_rate, decay_rate, decay_steps, momentum=0):
        if not hasattr(self, 'loss'):
            raise Exception('loss has not been defined, please run build_loss first')

        with tf.name_scope('train'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                    learning_rate, global_step, decay_steps, decay_rate, staircase=True)
                tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = self.optimizer.compute_gradients(self.loss)
            op_gradients = self.optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    pass
                    # tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                self.train_step = tf.identity(learning_rate, name='control')

    def build_saver(self, max_to_keep=100):
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    def build_summary(self, log_dir, sess: tf.Session):
        self.summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(log_dir, sess.graph)

    def initialize(self, sess: tf.Session):
        self.initializer = tf.global_variables_initializer()
        sess.run(self.initializer)

    def train(self, n_epochs, dropout, eval_freq, save_dir, sess: tf.Session,
              train_loader, val_loader):

        if not hasattr(self, 'loss'):
            raise Exception('loss has not been defined, please run build_loss first')
        if not hasattr(self, 'train_step'):
            raise Exception('train_step has not been defined, please run build_train first')

        print('training started')

        t_wall, t_interval = time.time(), time.time()
        batch_size = train_loader.batch_size
        aucs = []
        losses = []
        step = 1
        n_steps = train_loader.n_batches * n_epochs

        max_auc = 0.0
        last_max_ckpt = None

        for epoch in range(n_epochs):

            train_loader.async_load_batch(0)

            # train epoch
            for batch in range(train_loader.n_batches):

                t1 = time.time()
                batch_text, batch_image, labels = train_loader.get_async_loaded()
                if batch + 1 < train_loader.n_batches:
                    train_loader.async_load_batch(batch+1)
                # batch_text, batch_image, labels = train_loader.get_batch_by_index(batch)
                t2 = time.time()

                feed_dict = {
                    self.ph_labels: labels,
                    self.ph_dropout: dropout
                }
                for ph, data in zip(self.ph1, batch_text):
                    feed_dict[ph] = data
                for ph, data in zip(self.ph2, batch_image):
                    feed_dict[ph] = data

                learning_rate, loss_average, summary = sess.run(
                    [self.train_step, self.loss_average, self.summary],
                    feed_dict=feed_dict)
                if hasattr(self, 'summary'): self.writer.add_summary(summary, step)
                t3 = time.time()

                if step % eval_freq == 0:

                    val_report, val_auc, val_loss = self.evaluate(val_loader, sess)
                    aucs.append(val_auc)
                    losses.append(val_loss)

                    t_interval_1 = time.time()
                    eta = 1.0 * (t_interval_1 - t_interval) / eval_freq * (n_steps - step) / 3600
                    t_interval = t_interval_1
                    print('epoch %d/%d batch %d/%d progress: %.2f%% eta: %.2fhrs' % (epoch+1, n_epochs, batch + 1, train_loader.n_batches, 100.0*step/n_steps, eta))
                    print('    learning_rate: %.2e loss_average: %.2e' % (learning_rate, loss_average))
                    print('    batch_time: %.2fs, wait for data %.2fs, train step %.2fs, %.1fms per sample' %
                          (t3-t1, t2-t1, t3-t2, 1000*(t3-t1)/batch_size))
                    print('    validation: %s' % val_report)

                    summary = tf.Summary()
                    summary.ParseFromString(sess.run(self.summary, feed_dict))
                    summary.value.add(tag='validation/auc', simple_value=val_auc)
                    summary.value.add(tag='validation/loss', simple_value=val_loss)
                    self.writer.add_summary(summary, step)

                    if val_auc > max_auc:
                        if last_max_ckpt is not None:
                            os.system("rm %s*" % os.path.join(save_dir, last_max_ckpt))
                        max_auc = val_auc
                        last_max_ckpt = '%d_%.3f' % (epoch+1, val_auc)
                        self.save(os.path.join(save_dir, last_max_ckpt), sess)

                step += 1

            if (epoch + 1) % 1 == 0 or (epoch + 1) == n_epochs:
                self.save(os.path.join(save_dir, str(epoch+1)), sess)

        t_all = time.time() - t_wall
        t_step = t_all / step
        print('total %d steps in %.2fhrs, %.2fs per step' % (step, t_all / 3600, t_step))
        print('training finished, validation AUC peak = {:.2f}, mean = {:.2f}'.format(max(aucs), np.mean(aucs[-10:])))
        return aucs, losses

    def evaluate(self, loader, sess: tf.Session):

        if not hasattr(self, 'loss'):
            raise Exception('loss has not been defined, please run build_loss first')

        t_wall = time.time()

        total_loss = 0
        labels = []
        predictions = []

        loader.async_load_batch(0)

        for batch in range(loader.n_batches):
            batch_data_1, batch_data_2, batch_labels = loader.get_async_loaded()
            if batch + 1 < loader.n_batches: loader.async_load_batch(batch + 1)
            feed_dict = {
                self.ph_labels: batch_labels,
                self.ph_dropout: 1
            }
            for ph, data in zip(self.ph1, batch_data_1):
                feed_dict[ph] = data
            for ph, data in zip(self.ph2, batch_data_2):
                feed_dict[ph] = data

            batch_pred, batch_loss = sess.run([self.logits, self.loss], feed_dict=feed_dict)
            labels.append(batch_labels)
            predictions.append(batch_pred)
            total_loss += batch_loss

        labels = np.concatenate(labels, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        loss = total_loss / loader.n_batches

        fpr, tpr, thresholds = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        report = 'samples: {:d}, AUC : {:.3f}, loss: {:.4e} '.format(len(labels), roc_auc, loss)
        report += 'time: {:.1f}s'.format(time.time() - t_wall)

        return report, roc_auc, loss

    def save(self, path, sess: tf.Session):
        self.saver.save(sess, path)

    def restore(self, ckpt_path, sess:tf.Session):
        self.saver.restore(sess, ckpt_path)
