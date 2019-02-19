from lib.mlnet import MLNet
from data.data_loader import DataLoader
from data.utils import split_and_pack
import tensorflow as tf
import numpy as np
import time

def get_descs_and_labels(net: MLNet, sess: tf.Session, modal,
                         paths_with_labels, process_fn, batch_size):

    if net.is_training: raise Exception("should not run this in training mode")
    if net.is_retrieving: raise Exception("should not run this in retrieving mode")

    descriptors = []
    labels = []

    loader = DataLoader(paths_with_labels, batch_size, shuffle=False, process_fn=process_fn)

    for batch in range(loader.n_batches):

        batch_data, batch_labels = loader.get_batch_by_index(batch)
        batch_data = split_and_pack(batch_data)

        if modal == 1:
            feed_dict = {}
            for ph, data in zip(net.ph1, batch_data):
                feed_dict[ph] = data
            batch_descs = net.descriptors_1.eval(session=sess, feed_dict=feed_dict)
        elif modal == 2:
            feed_dict = {}
            for ph, data in zip(net.ph2, batch_data):
                feed_dict[ph] = data
            batch_descs = net.descriptors_2.eval(session=sess, feed_dict=feed_dict)
        else:
            raise Exception("modal should be either 1 or 2")

        descriptors.append(batch_descs)
        labels.append(batch_labels)

    if loader.n_remain > 0:
        batch_data, batch_labels = loader.get_remaining()
        batch_data = split_and_pack(batch_data)

        if modal == 1:
            feed_dict = {}
            for ph, data in zip(net.ph1, batch_data):
                feed_dict[ph] = data
            batch_descs = net.descriptors_1.eval(session=sess, feed_dict=feed_dict)
        elif modal == 2:
            feed_dict = {}
            for ph, data in zip(net.ph2, batch_data):
                feed_dict[ph] = data
            batch_descs = net.descriptors_2.eval(session=sess, feed_dict=feed_dict)
        else:
            raise Exception("modal should be either 1 or 2")

        descriptors.append(batch_descs[:loader.n_remain])
        labels.append(batch_labels[:loader.n_remain])

    descriptors = np.concatenate(descriptors, axis=0)
    labels = np.concatenate(labels, axis=0)

    return descriptors, labels


def retrieve(net: MLNet, sess: tf.Session,
             q_desc, q_label, r_descs, r_labels,
             at=100, batch_size=128):

    if not net.is_retrieving: raise Exception("should run this in retrieving mode")

    n_entries = len(r_descs)
    desc_dims = len(q_desc)
    n_batches = int(n_entries / batch_size)
    n_remain = n_entries % batch_size

    logits = []
    labels = []

    batch_q_descs = np.repeat(np.expand_dims(q_desc, axis=0), batch_size, axis=0)
    batch_q_labels = np.array([q_label for _ in range(batch_size)], dtype='int32')

    for batch in range(n_batches):
        batch_r_descs = r_descs[batch * batch_size: (batch + 1) * batch_size]
        batch_r_labels = r_labels[batch * batch_size:(batch + 1) * batch_size]
        batch_labels = np.array(batch_q_labels == batch_r_labels, dtype='int32')
        feed_dict = {net.ph_desc_1: batch_q_descs, net.ph_desc_2: batch_r_descs}
        batch_logits = net.logits.eval(session=sess, feed_dict=feed_dict)
        logits.append(batch_logits)
        labels.append(batch_labels)

    if n_remain > 0:
        batch_r_descs = np.zeros([batch_size, desc_dims], dtype='float32')
        batch_r_descs[:n_remain, :] = r_descs[-n_remain:]
        batch_r_labels = np.zeros([batch_size], dtype='int32')
        batch_r_labels[:n_remain] = r_labels[-n_remain:]
        batch_labels = np.array(batch_q_labels == batch_r_labels, dtype='int32')

        feed_dict = {net.ph_desc_1: batch_q_descs, net.ph_desc_2: batch_r_descs}
        batch_logits = net.logits.eval(session=sess, feed_dict=feed_dict)
        logits.append(batch_logits[:n_remain])
        labels.append(batch_labels[:n_remain])

    indices = [i for i in range(n_entries)]
    logits = np.concatenate(logits, axis=0).tolist()
    labels = np.concatenate(labels, axis=0).tolist()
    zipped = list(zip(indices, logits, labels))
    zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    indices, logits, labels = zip(*zipped)

    n_relavant = 0
    precisions = []
    piv = len(labels) if at <= 0 or at > len(labels) else at
    for j in range(piv):
        if labels[j] == 1:
            n_relavant += 1
            precisions.append(1.0 * n_relavant / (j + 1))

    if n_relavant == 0: precisions = [0]

    average_precision = sum(precisions) / len(precisions)

    return indices[:at], average_precision

def average_precisions(net: MLNet, sess: tf.Session,
                       q_descs, q_labels, r_descs, r_labels,
                       at=100, batch_size=128):

    """
    :param net: an MLNet model
    :param sess: a tensorflow session=
    :param q_descs: descriptors for querying data
    :param q_labels: labels for querying data
    :param r_descs: descriptors for retrieved data
    :param r_labels: labels for retrieved data
    :param at: if mAP@100 is desired, assign at with 100, if mAP@ALL is desired, assign at with 0
    :param batch_size: batch size
    :return: average procisions
    """

    n_samples, n_entries = len(q_descs), len(r_descs)

    APs = []

    for query_idx in range(n_samples):
        time1 = time.time()
        _, average_precision = retrieve(net, sess, q_descs[query_idx], q_labels[query_idx], r_descs, r_labels, at=at, batch_size=batch_size)
        APs.append(average_precision)
        time2 = time.time()
        ellapsed = time2 - time1
        print("sample %4d/%4d, AP: %5.3f, time: %5.2fs" %
                 (query_idx + 1, n_samples, average_precision, ellapsed), end='\r')

    return APs
