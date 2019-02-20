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

# parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--sess", help="session name, such as cmplaces1_1536482758")
parser.add_argument("--ckpt", help="checkpoint to restore")
parser.add_argument("--epochs", help="number of epochs")
parser.add_argument("--bsize", help="batch size")
parser.add_argument("--lthreads", help="number of loader threads")
parser.add_argument("--lswone", help="if labels start with one instead of zero, add this flag",
                    action="store_true")
parser.add_argument("--m1device", help="specify modal 1 device. e.g.: cpu0, gpu1, ...")
parser.add_argument("--m2device", help="specify modal 2 device. e.g.: cpu0, gpu1, ...")
parser.add_argument("--mtrdevice", help="specify metrics device. e.g.: cpu0, gpu1, ...")
args = parser.parse_args()

sess_name = args.sess if args.sess else "skr"
ckpt_name = args.ckpt if args.sess else None
n_epochs = args.epochs if args.epochs else 50
batch_size = int(args.bsize) if args.bsize else 128
n_loader_threads = int(args.lthreads) if args.lthreads else 8
label_start_with_zero = False if args.lswone else True

modal_1_device = parse_device_str(args.m1device) if args.m1device else '/device:GPU:0'
modal_2_device = parse_device_str(args.m1device) if args.m2device else '/device:GPU:0'
metrics_device = parse_device_str(args.mtrdevice) if args.mtrdevice else '/device:GPU:0'

# initialize data loader

text_dir = '/home1/yul/ych/text_processing/exp_data'
image_dir = '/home1/yul/yzq/data/cmplaces/natural_vgg19_relu7'

text_train_path = '/home1/yul/ych/text_processing/text_train_list.txt'
image_train_path = '/home1/yul/yzq/data/cmplaces/natural_vgg19_relu7_train.txt'
text_val_path = '/home1/yul/ych/text_processing/text_val_list.txt'
image_val_path = '/home1/yul/yzq/data/cmplaces/natural_vgg19_relu7_val.txt'

adjmat_path = '/home1/yul/ych/text_processing/exp_data/graph_knn_kg.txt'
n_classes = 205
label_start_with_zero = True
n_train = 204800
n_val = 8192

text_train_list = parse_data_file_list(text_dir, text_train_path, label_start_with_zero)
image_train_list = parse_data_file_list(image_dir, image_train_path, label_start_with_zero)
text_val_list = parse_data_file_list(text_dir, text_val_path, label_start_with_zero)
image_val_list = parse_data_file_list(image_dir, image_val_path, label_start_with_zero)

train_loader = PosNegLoader(text_train_list, image_train_list,
                            n_train, n_train,
                            batch_size=batch_size, n_classes=n_classes,
                            shuffle=True, n_threads=n_loader_threads,
                            process_fn_1=None, process_fn_2=None)

val_loader = PosNegLoader(text_val_list, image_val_list,
                          n_val, n_val,
                          batch_size=batch_size, n_classes=n_classes,
                          shuffle=True, n_threads=n_loader_threads,
                          process_fn_1=None, process_fn_2=None)

# build network, loss, train

desc_dims = 1024
out_dims = 1
net = Model(adjmat_path, batch_size, desc_dims, out_dims, is_training=True)
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

n_epochs = 50
eval_freq = 400
dropout = 0.6
out_dir = os.path.join('..', '..', 'out', sess_name)

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
    if ckpt_name is not None:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        net.restore(ckpt_path, sess)
    net.train(n_epochs, dropout, eval_freq, ckpt_dir, sess, train_loader, val_loader)
