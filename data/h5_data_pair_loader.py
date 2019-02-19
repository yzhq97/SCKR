import time
import random
from data.h5_data_loader import DataLoader
from data.utils import *
from abc import ABCMeta, abstractmethod
import threading

class DataPairLoader:

    __metaclass__ = ABCMeta  # this class is an abstract class

    def __init__(self, h5_path_1, h5_path_2, split_1, split_2,
                 batch_size, n_classes, shuffle=False, whole_batches=False):

        random.seed(int(1e6 * (time.time() % 1)))

        self.loader_1 = DataLoader(h5_path_1, batch_size, split=split_1, whole_batches=whole_batches)
        self.loader_2 = DataLoader(h5_path_2, batch_size, split=split_2, whole_batches=whole_batches)

        self.indices_1 = [_ for _ in self.loader_1.indices]
        self.indices_2 = [_ for _ in self.loader_2.indices]

        # parameters
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_samples_1 = len(self.indices_1)
        self.n_samples_2 = len(self.indices_2)

        # generate data pairs
        indices_of_indices_1, indices_of_indices_2 = self.generate_pair_indices()
        if shuffle:
            zipped = list(zip(indices_of_indices_1, indices_of_indices_2))
            random.shuffle(zipped)
            indices_of_indices_1, indices_of_indices_2 = zip(*zipped)

        loader_1_indices = [self.indices_1[i] for i in indices_of_indices_1]
        loader_2_indices = [self.indices_2[i] for i in indices_of_indices_2]

        # initialize loaders
        self.loader_1.set_indices(loader_1_indices)
        self.loader_2.set_indices(loader_2_indices)

        # state
        self.n_pairs = len(indices_of_indices_1)
        if whole_batches:
            self.n_batches = math.floor(self.n_pairs / self.batch_size)
            self.n_remain = self.n_pairs % batch_size
        else:
            self.n_batches = math.ceil(self.n_pairs / self.batch_size)
            self.n_remain = 0
        self.i = 0

        # async_load
        self.async_load_pool = [None, None, None]
        self.async_load_thread = None

    @abstractmethod
    def generate_pair_indices(self):
        return [], []

    def reset(self):
        self.i = 0
        self.loader_1.reset()
        self.loader_2.reset()

    def set_batch_index(self, i):
        self.i = i
        self.loader_1.set_batch_index(i)
        self.loader_2.set_batch_index(i)

    def get_batch_by_index(self, i):
        data_1, labels_1 = self.loader_1.get_batch_by_index(i)
        data_2, labels_2 = self.loader_2.get_batch_by_index(i)
        labels = np.array(labels_1 == labels_2, dtype="int32")
        return data_1, data_2, labels

    def get_remaining(self):
        data_1, labels_1 = self.loader_1.get_remaining()
        data_2, labels_2 = self.loader_2.get_remaining()
        labels = np.array(labels_1 == labels_2, dtype="int32")
        return data_1, data_2, labels

    def next_batch(self):
        if self.i < self.n_batches:
            data_1, data_2, labels = self.get_batch_by_index(self.i)
            self.i += 1
            return data_1, data_2, labels
        else:
            return [], [], None

    def get_pair_by_index(self, i):
        datup_1, label_1 = self.loader_1.get_datup_at_index(i)
        datup_2, label_2 = self.loader_2.get_datup_at_index(i)
        label = int(label_1==label_2)
        return datup_1, datup_2, int(label)

    def async_load_batch(self, i):
        if self.async_load_thread is not None:
            self.async_load_thread.join()
        self.async_load_thread = AsyncLoadThread(self, i)
        self.async_load_thread.start()

    def get_async_loaded(self):
        if self.async_load_thread is None: raise Exception('Did not load anything')
        self.async_load_thread.join()
        data_1, data_2, labels = self.async_load_pool
        return data_1, data_2, labels

class PosNegLoader(DataPairLoader):
    """
    This DataPairLoader generates postive and negative sample pairs for training
    """

    def __init__(self, h5_path_1, h5_path_2, split_1, split_2, n_pos, n_neg,
                 batch_size, n_classes, shuffle=False, whole_batches=False):

        self.n_pos = n_pos
        self.n_neg = n_neg

        DataPairLoader.__init__(self, h5_path_1, h5_path_2, split_1, split_2,
                                batch_size, n_classes, shuffle, whole_batches)


    def generate_pair_indices(self):
        n_pos, n_neg = self.n_pos, self.n_neg
        y1 = self.loader_1.ds_labels[self.indices_1]
        y2 = self.loader_2.ds_labels[self.indices_2]

        pos_indices = []
        neg_indices = []

        # positive pairs
        for i in range(self.n_classes):
            opt1 = np.where(y1 == i)[0]
            opt2 = np.where(y2 == i)[0]
            n1 = len(opt1)
            n2 = len(opt2)
            n_need = int(n_pos / self.n_classes)
            if n1 != 0 and n1 != 0:
                opt_tups = [(opt1[i], opt2[j]) for i in range(n1) for j in range(n2)]
                if len(opt_tups) < n_need:
                    cat_indices = opt_tups[:]
                    while len(cat_indices) < n_need:
                        cat_indices.append(random.choice(opt_tups))
                    pos_indices.extend(cat_indices)
                else:
                    cat_indices = random.sample(opt_tups, n_need)
                    pos_indices.extend(cat_indices)
            else:
                print('data for class %d is missing' % i)

        # negative pairs
        n1, n2 = len(y1), len(y2)
        while len(neg_indices) < n_neg:
            idx1 = random.randint(1, n1) - 1
            idx2 = random.randint(1, n2) - 1
            tup = (idx1, idx2)
            if y1[idx1] != y2[idx2]:
                neg_indices.append(tup)

        indices = pos_indices + neg_indices
        indices_1 = [tup[0] for tup in indices]
        indices2 = [tup[1] for tup in indices]

        return indices_1, indices2


class MAPLoader(DataPairLoader):
    """
    This DataLoader generates sample pairs in a traversal fashion, used is mean_average_precision calculation.
    """

    def __init__(self, h5_path_1, h5_path_2, split_1, split_2, query_idx, query_modal,
                 batch_size, n_classes, shuffle=False, whole_batches=False):

        self.query_idx = query_idx
        self.query_modal = query_modal

        DataPairLoader.__init__(self, h5_path_1, h5_path_2, split_1, split_2,
                                batch_size, n_classes, shuffle, whole_batches)

    def generate_pair_indices(self):
        if self.query_modal == 1:
            indices_1 = [self.i] * self.n_samples_2
            indices_2 = [j for j in range(self.n_samples_2)]
            return indices_1, indices_2
        elif self.query_modal == 2:
            indices_1 = [j for j in range(self.n_samples_1)]
            indices_2 = [self.i] * self.n_samples_1
            return indices_1, indices_2
        else:
            raise Exception("query_modal number should be 1 or 2")

class AsyncLoadThread(threading.Thread):
    def __init__(self, master, batch_idx):
        threading.Thread.__init__(self)
        self.master = master
        self.batch_idx = batch_idx

    def run(self):
        if self.batch_idx == self.master.n_batches and self.master.n_remain > 0:
            self.master.async_load_pool = self.master.get_remaining()
        elif 0 <= self.batch_idx < self.master.n_batches:
            self.master.async_load_pool = self.master.get_batch_by_index(self.batch_idx)
        else:
            raise Exception("Invalid batch index!")
