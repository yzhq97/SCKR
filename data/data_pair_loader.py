import time
import random
from data.data_loader import DataLoader
from data.utils import *
from abc import ABCMeta, abstractmethod
import threading

class DataPairLoader:
    """
    This is an abstract class
    DataPairLoader accepts two lists of data files and allows the training/testing program to get data by batches according to some order
    How the data are paired depends on the implementation of the abstract method generate_pair_indices.
    """

    __metaclass__ = ABCMeta  # this class is an abstract class

    def __init__(self,
                 paths_with_labels_1, paths_with_labels_2,
                 batch_size, n_classes, shuffle, n_threads=8,
                 process_fn_1=None, process_fn_2=None):

        random.seed(int(1e6 * (time.time() % 1)))

        self.paths_with_labels_1 = paths_with_labels_1
        self.paths_with_labels_2 = paths_with_labels_2

        # parameters
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.n_samples_1 = len(self.paths_with_labels_1)
        self.n_samples_2 = len(self.paths_with_labels_2)
        self.dtype_labels = 'int32'

        # generate data pairs
        indices_1, indices_2 = self.generate_pair_indices()
        if shuffle:
            indices = list(zip(indices_1, indices_2))
            random.shuffle(indices)
            indices_1, indices_2 = zip(*indices)

        loader_1_list = [self.paths_with_labels_1[i] for i in indices_1]
        loader_2_list = [self.paths_with_labels_2[i] for i in indices_2]

        # initialize loaders
        self.loader_1 = DataLoader(loader_1_list,
                                   batch_size=self.batch_size,
                                   n_threads=n_threads,
                                   process_fn=process_fn_1)
        self.loader_2 = DataLoader(loader_2_list,
                                   batch_size=self.batch_size,
                                   n_threads=n_threads,
                                   process_fn=process_fn_2)

        # state
        self.n_pairs = len(indices_1)
        self.n_batches = math.floor(self.n_pairs / self.batch_size)
        self.n_remain = self.n_pairs % batch_size
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
        labels = (labels_1 == labels_2).astype(self.dtype_labels)
        data_1 = split_and_pack(data_1)
        data_2 = split_and_pack(data_2)
        return data_1, data_2, labels

    def get_remaining(self):
        data_1, labels_1 = self.loader_1.get_remaining()
        data_2, labels_2 = self.loader_2.get_remaining()
        labels = (labels_1 == labels_2).astype(self.dtype_labels)
        data_1 = split_and_pack(data_1)
        data_2 = split_and_pack(data_2)
        return data_1, data_2, labels

    def next_batch(self):
        if self.i < self.n_batches:
            data_1, data_2, labels = self.get_batch_by_index(self.i)
            self.i += 1
            data_1 = split_and_pack(data_1)
            data_2 = split_and_pack(data_2)
            return data_1, data_2, labels
        else:
            return [], [], None

    def get_pair_by_index(self, i):
        datup_1, label_1 = self.loader_1.get_datup_at_index(i)
        datup_2, label_2 = self.loader_2.get_datup_at_index(i)
        label = label_1==label_2
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

    def __init__(self,
                 paths_with_labels_1, paths_with_labels_2,
                 n_pos, n_neg,
                 batch_size, n_classes, shuffle=False, n_threads=8,
                 process_fn_1=None, process_fn_2=None):

        self.n_pos = n_pos
        self.n_neg = n_neg

        DataPairLoader.__init__(self,
                                paths_with_labels_1, paths_with_labels_2,
                                batch_size, n_classes, shuffle, n_threads,
                                process_fn_1, process_fn_2)


    def generate_pair_indices(self):
        n_pos, n_neg = self.n_pos, self.n_neg
        y1 = np.array([label for data_path, label in self.paths_with_labels_1])
        y2 = np.array([label for data_path, label in self.paths_with_labels_2])

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
            if y1[idx1] != y2[idx2] and tup not in neg_indices:
                neg_indices.append(tup)

        indices = pos_indices + neg_indices
        indices1 = [tup[0] for tup in indices]
        indices2 = [tup[1] for tup in indices]

        return indices1, indices2


class MAPLoader(DataPairLoader):
    """
    This DataLoader generates sample pairs in a traversal fashion, used is mean_average_precision calculation.
    """

    def __init__(self,
                 paths_with_labels_1, paths_with_labels_2,
                 query_idx, query_modal, batch_size, n_classes, shuffle, n_threads=8,
                 process_fn_1=None, process_fn_2=None):

        """
        :param query_idx: The index of query data
        :param query_modal: should be 1 or 2. indicates which modal is query and which modal is retrieved
        e.g.: query_idx=4, query_modal=2, then the traversed indices pairs would be:
        [ (0, 4), (1, 4), (2, 4), (3, 4), (5, 4), (6, 4), (7, 4), ...]
        :param label_start_with_zero: set to True if input labels are 1, 2, 3... set to False if input labels are 0, 1, 2 ...
        """

        self.i = query_idx
        self.query_modal = query_modal

        DataPairLoader.__init__(self,
                                paths_with_labels_1, paths_with_labels_2,
                                batch_size, n_classes, shuffle, n_threads,
                                process_fn_1, process_fn_2)

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
