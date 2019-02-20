import sys
sys.path.append('../..')
import argparse
import os
from data.utils import parse_data_file_list
from lib.eval import get_descs_and_labels, average_precisions
import random
import tensorflow as tf
import numpy as np
from models.cmplaces.sckr import Model

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="path to the directory under which data is placed")
    parser.add_argument("--model", help="'sr', 'scr', 'skr' or 'sckr'", default="sckr")
    parser.add_argument("--sess", help="session name, such as sckr_1536482758", default="default_session")
    parser.add_argument("--ckpts", help="comma seperated checkpoint names, such as 10,20,30,40,50", default='10,20,30,40,50')
    parser.add_argument("--samples", help="number of test samples", default=1000)
    parser.add_argument("--lswone", help="if labels start with one instead of zero, add this flag",
                        action="store_true")
    parser.add_argument("--bsize", help="batch size", default=128)
    parser.add_argument("--args.desc_dims", help="number of dimensions of description vectors", default=1024)

    args = parser.parse_args()

    args.ckpts = args.ckpts.split(',')
    args.lswone = False if args.lswone else True

    return args

def test_ckpt(ckpt_path, q1_list, r1_list, q2_list, r2_list):

    # get descriptors

    net = Model(adjmat_path, args.bsize, args.desc_dims, 1)
    net.build()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        net.restore(ckpt_path, sess)
        q1_descs, q1_labels = get_descs_and_labels(net, sess, 1, q1_list, None, args.bsize)
        q2_descs, q2_labels = get_descs_and_labels(net, sess, 2, q2_list, None, args.bsize)
        r1_descs, r1_labels = get_descs_and_labels(net, sess, 1, r1_list, None, args.bsize)
        r2_descs, r2_labels = get_descs_and_labels(net, sess, 2, r2_list, None, args.bsize)

    r1_indices = [i for i in range(len(r1_list))]
    r1s_indices = random.sample(r1_indices, args.samples)
    r1s_descs = r1_descs[r1s_indices]
    r1s_labels = r1_labels[r1s_indices]

    r2_indices = [i for i in range(len(r2_list))]
    r2s_indices = random.sample(r2_indices, args.samples)
    r2s_descs = r2_descs[r2s_indices]
    r2s_labels = r2_labels[r2s_indices]

    del net
    tf.reset_default_graph()

    # retrieval

    net = Model(adjmat_path, args.bsize, args.desc_dims, 1, is_retrieving=True)
    net.build()
    with tf.Session(config=config) as sess:
        net.restore(ckpt_path, sess)

        APs_1 = average_precisions(net, sess,
                                   q1_descs, q1_labels,
                                   r2_descs, r2_labels,
                                   100)
        mAP1 = sum(APs_1) / len(APs_1)

        APs_2 = average_precisions(net, sess,
                                   q2_descs, q2_labels,
                                   r1_descs, r1_labels,
                                   100, args.bsize)
        mAP2 = sum(APs_2) / len(APs_2)

        APs_3 = average_precisions(net, sess,
                                   r1s_descs, r1s_labels,
                                   r2_descs, r2_labels,
                                   100, args.bsize)
        mAP3 = sum(APs_3) / len(APs_3)

        APs_4 = average_precisions(net, sess,
                                   r2s_descs, r2s_labels,
                                   r1_descs, r1_labels,
                                   100, args.bsize)
        mAP4 = sum(APs_4) / len(APs_4)

    del net
    tf.reset_default_graph()

    return mAP1, mAP2, mAP3, mAP4

if __name__ == "__main__":

    args = parse_args()

    text_dir = os.path.join(args.data_dir, 'text_bow')
    image_dir = os.path.join(args.data_dir, 'natural_vgg19_relu7')

    text_train_path = os.path.join(args.data_dir, 'text_train_list')
    image_train_path = os.path.join(args.data_dir, 'natural_vgg19_relu7_train.txt')
    text_val_path = os.path.join(args.data_dir, 'text_val_list')
    image_val_path = os.path.join(args.data_dir, 'natural_vgg19_relu7_val.txt')
    text_test_path = os.path.join(args.data_dir, 'text_test_list.txt')
    image_test_path = os.path.join(args.data_dir, 'natural_vgg19_relu7_test.txt')

    adjmat_path = os.path.join(args.data_dir, 'graph', '%s.txt' % args.model)

    out_dir = os.path.join('..', '..', 'out', args.sess)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')

    text_train_list = parse_data_file_list(text_dir, text_train_path, args.lswone)
    image_train_list = parse_data_file_list(image_dir, image_train_path, args.lswone)
    text_val_list = parse_data_file_list(text_dir, text_val_path, args.lswone)
    image_val_list = parse_data_file_list(image_dir, image_val_path, args.lswone)
    text_test_list = parse_data_file_list(text_dir, text_test_path, args.lswone)
    image_test_list = parse_data_file_list(image_dir, image_test_path, args.lswone)

    if __name__ == '__main__':
        print("running validation")
        results = []
        for ckpt_name in args.ckpts:
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            mAP1, mAP2, mAP3, mAP4 = test_ckpt(ckpt_path, text_test_list, text_train_list, image_test_list, image_train_list)
            results.append((ckpt_name, mAP1, mAP2, (mAP1 + mAP2) / 2, mAP3, mAP4, (mAP3 + mAP4) / 2))
            print("    %10s | %6.4f, %6.4f, %6.4f | %6.4f, %6.4f, %6.4f" %
                  (ckpt_name, mAP1, mAP2, (mAP1 + mAP2) / 2, mAP3, mAP4, (mAP3 + mAP4) / 2))

        results = sorted(results, key=lambda x: x[3], reverse=True)
        best = results[0]
        ckpt_path = os.path.join(ckpt_dir, best[0])
        mAP1, mAP2, mAP3, mAP4 = test_ckpt(ckpt_path, text_test_list, text_train_list, image_test_list, image_train_list)
        print("best model validation results:")
        print("    %6s | %6.4f, %6.4f, %6.4f | %6.4f, %6.4f, %6.4f" %
              (best[0], best[1], best[2], best[3], best[4], best[5], best[6]))
        print("best model test results:")
        print("    %6s | %6.4f, %6.4f, %6.4f | %6.4f, %6.4f, %6.4f" %
              (best[0], mAP1, mAP2, (mAP1 + mAP2) / 2, mAP3, mAP4, (mAP3 + mAP4) / 2))

