import numpy as np
import os
from data.utils import *
import threading
import random
import cv2
import time

class DataLoader:
    """
    DataLoader accepts a list of data files and allows the training/testing program to get data by batches
    for more information, checkout files in the doc folder
    """

    def __init__(self, paths_with_labels, batch_size, n_threads=8,
                 shuffle=False, process_fn=None, dtype_labels='int32'):
        """
        :param paths_with_labels: a list like [ (file_path, label) ]
        :param batch_size: batch size
        :param n_threads: number of threads used when loading and preprocessing files
        :param shuffle: whether or not to shuffle data
        :param process_fn: a function which takes in a path of a single data file and returns desired preprocessed data
        :param dtype_labels: data type of the labels
        """
        random.seed(int(1e6 * (time.time() % 1)))

        self.paths_with_labels = paths_with_labels
        if shuffle: random.shuffle(self.paths_with_labels)

        # parameters
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.process_fn = process_fn
        self.dtype_labels = dtype_labels

        # threading
        self.lock = threading.Lock()
        self.thread_pool = []
        self.data_deliver_pool = [] # where DataLoaderThreads deliver loaded data
        self.labels_deliver_pool = []

        # state
        self.n_samples = len(self.paths_with_labels)
        self.n_batches = math.floor(self.n_samples / self.batch_size)
        self.n_remain = self.n_samples % batch_size
        self.i = 0

    def reset(self):
        self.i = 0

    def get_data_by_list(self, data_list):
        divided_lists = divide_list(data_list, self.n_threads)

        self.thread_pool = []
        self.data_deliver_pool = [[] for i in range(self.n_threads)]
        self.labels_deliver_pool = [None for i in range(self.n_threads)]

        self.thread_pool = [DataLoaderThread(self, i, divided_lists[i]) for i in range(self.n_threads)]
        for i in range(self.n_threads): self.thread_pool[i].start()
        for i in range(self.n_threads): self.thread_pool[i].join()

        data = [datup for pool in self.data_deliver_pool for datup in pool]
        labels = [l for l in self.labels_deliver_pool if l is not None]

        labels = np.concatenate(labels, axis=0).astype(self.dtype_labels)

        return data, labels

    def set_batch_index(self, i):
        self.i = i

    def get_batch_by_index(self, i):
        if i >= self.n_batches:
            raise Exception("batch index exceeds limit")

        start = i * self.batch_size
        end = (i+1) * self.batch_size

        batch_list = self.paths_with_labels[start:end]

        return self.get_data_by_list(batch_list)

    def get_remaining(self):
        if self.n_remain == 0: return None

        start = self.n_samples - self.n_remain
        end = self.n_samples

        batch_list = self.paths_with_labels[start:end]
        data, labels = self.get_data_by_list(batch_list)

        fake_data = [data[0] for i in range(len(data), self.batch_size)]
        fake_labels = [labels[0] for i in range(len(labels), self.batch_size)]
        fake_labels = np.array(fake_labels)

        data = data + fake_data
        labels = np.concatenate([labels, fake_labels], axis=0)

        return data, labels

    def get_datup_at_index(self, i):
        file_path, label = self.paths_with_labels[i]

        if self.process_fn is not None:
            datup = self.process_fn(file_path)
            if not isinstance(datup, np.ndarray):
                raise Exception("value returned by process_fn should be a numpy.ndarray")
        else:
            _, extension = os.path.splitext(file_path)
            extension = extension.lower()
            if extension == '.npy':
                datup = [np.load(file_path)]
            elif extension in image_extensions:
                img = cv2.imread(file_path)
                if len(img.shape) == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                datup = [img]
            else:
                errstr = "process_fn should be provided for %s files" % extension
                raise Exception(errstr)

        return datup, label

    def next_batch(self):
        if self.i < self.n_batches:
            data, labels = self.get_batch_by_index(self.i)
            self.i += 1
            return data, labels
        else:
            return None, None

    def deliver(self, i, data, labels):
        self.data_deliver_pool[i] = data
        self.labels_deliver_pool[i] = labels

class DataLoaderThread(threading.Thread):

    def __init__(self, master, id, paths_with_labels):
        threading.Thread.__init__(self)
        self.master = master
        self.id = id
        self.paths_with_labels = paths_with_labels

    def run(self):
        if len(self.paths_with_labels) == 0: return

        data = []
        labels = []

        for file_path, label in self.paths_with_labels:

            if self.master.process_fn is not None:
                datup = self.master.process_fn(file_path)
            else:
                _, extension = os.path.splitext(file_path)
                extension = extension.lower()
                if extension == '.npy':
                    datup = [np.load(file_path)]
                elif extension in image_extensions:
                    img = cv2.imread(file_path)
                    if len(img.shape) == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    datup = [img]
                else:
                    errstr = "process_fn should be provided for %s files" % extension
                    raise Exception(errstr)

            data.append(datup)
            labels.append(label)

        labels = np.array(labels)

        self.master.lock.acquire()
        self.master.deliver(self.id, data, labels)
        self.master.lock.release()