import sys
sys.path.append('../..')
from models.cmplaces.sckr import Model
from data.data_pair_loader import PosNegLoader
from data.utils import parse_data_file_list
from main.utils import parse_device_str
import time
import os
import tensorflow as tf
import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="path to the directory under which data is placed")
    parser.add_argument("--model", help="'sr', 'scr', 'skr' or 'sckr'", default="sckr")
    parser.add_argument("--sess", help="session name, such as sckr_1536482758", default="default_session")
    parser.add_argument("--ckpt", help="checkpoint to restore", default=None)
    parser.add_argument("--epochs", help="number of epochs", default=50)
    parser.add_argument("--bsize", help="batch size", default=128)
    parser.add_argument("--desc_dims", help="number of dimensions of description vectors", default=1024)
    parser.add_argument("--n_pairs_train", help="number of positive/negative pairs to use for training", default=204800)
    parser.add_argument("--n_pairs_val", help="number of positive/negative pairs to use for validation", default=8192)
    parser.add_argument("--n_classes", help="number of total categories", default=205)
    parser.add_argument("--lthreads", help="number of loader threads", default=8)
    parser.add_argument("--lswone", help="if labels start with one instead of zero, add this flag",
                        action="store_true")
    parser.add_argument("--m1device", help="specify modal 1 device. e.g.: cpu0, gpu1, ...", default=parse_device_str('gpu0'))
    parser.add_argument("--m2device", help="specify modal 2 device. e.g.: cpu0, gpu1, ...", default=parse_device_str('gpu0'))
    parser.add_argument("--mtrdevice", help="specify metrics device. e.g.: cpu0, gpu1, ...", default=parse_device_str('gpu0'))
    args = parser.parse_args()

    args.lswone = False if args.lswone else True

    args.m1device = parse_device_str(args.m1device)
    args.m1device = parse_device_str(args.m1device)
    args.mtrdevice = parse_device_str(args.mtrdevice)

    return args

# initialize data loader

if __name__ == "__main__":

    args = parse_args()

    text_dir = os.path.join(args.data_dir, 'text_bow')
    image_dir = os.path.join(args.data_dir, 'natural_vgg19_relu7')

    text_train_path = os.path.join(args.data_dir, 'text_train_list')
    image_train_path = os.path.join(args.data_dir, 'natural_vgg19_relu7_train.txt')
    text_val_path = os.path.join(args.data_dir, 'text_val_list')
    image_val_path = os.path.join(args.data_dir, 'natural_vgg19_relu7_val.txt')

    adjmat_path = os.path.join(args.data_dir, 'graph', '%s.txt' % args.model)

    text_train_list = parse_data_file_list(text_dir, text_train_path, args.lswone)
    image_train_list = parse_data_file_list(image_dir, image_train_path, args.lswone)
    text_val_list = parse_data_file_list(text_dir, text_val_path, args.lswone)
    image_val_list = parse_data_file_list(image_dir, image_val_path, args.lswone)

    train_loader = PosNegLoader(text_train_list, image_train_list,
                                args.n_pairs_train, args.n_pairs_train,
                                batch_size=args.bsize, n_classes=args.n_classes,
                                shuffle=True, n_threads=args.lthreads,
                                process_fn_1=None, process_fn_2=None)

    val_loader = PosNegLoader(text_val_list, image_val_list,
                              args.n_pairs_val, args.n_pairs_val,
                              batch_size=args.bsize, n_classes=args.n_classes,
                              shuffle=True, n_threads=args.lthreads,
                              process_fn_1=None, process_fn_2=None)

    # build network, loss, train
    net = Model(adjmat_path, args.bsize, args.desc_dims, 1, is_training=True)
    net.build()

    lamda = 0.35
    mu = 0.8
    regularization = 5e-3
    net.build_loss(lamda, mu, regularization)

    learning_rate = 1e-4
    decay_rate = 0.95
    decay_steps = train_loader.n_batches
    net.build_train(learning_rate, decay_rate, decay_steps)

    # start training

    eval_freq = 400
    dropout = 0.6
    out_dir = os.path.join('..', '..', 'out', args.sess)

    log_dir = os.path.join(out_dir, 'log')
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    with tf.Session(config=config) as sess:
        net.build_summary(log_dir, sess)
        net.initialize(sess)
        if args.ckpt is not None:
            ckpt_path = os.path.join(ckpt_dir, args.ckpt)
            net.restore(ckpt_path, sess)
        net.train(args.epochs, dropout, eval_freq, ckpt_dir, sess, train_loader, val_loader)
