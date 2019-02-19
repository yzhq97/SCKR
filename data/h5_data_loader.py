from data.utils import *
import h5py
import random
import threading

class DataLoader:

    def __init__(self, h5_path, batch_size, split=None, whole_batches=False, shuffle=False):

        self.h5_path = h5_path
        self.batch_size = batch_size
        self.whole_batches = whole_batches
        self.shuffle = shuffle

        # load h5
        self.hdf = h5py.File(self.h5_path, 'r')

        self.g_data = self.hdf.get("data")
        self.g_splits = self.hdf.get("splits")
        self.ds_labels = self.hdf.get("labels")
        self.ds_origins = self.hdf.get("origins")
        self.ds_col_keys = self.hdf.get("col_keys")

        if split is not None:
            self.set_split(split)
        else:
            self.indices = [i for i in range(len(self.ds_labels))]

        self.col_keys = [ck.decode() for ck in self.ds_col_keys]
        self.n_cols = len(self.col_keys)

        # state

        self.n_samples = len(self.indices)
        if whole_batches:
            self.n_batches = math.floor(self.n_samples / self.batch_size)
            self.n_remain = self.n_samples % batch_size
        else:
            self.n_batches = math.ceil(self.n_samples / self.batch_size)
            self.n_remain = 0

        self.i = 0

    def reset(self):
        self.i = 0

    def get_data_by_indices(self, indices):
        data_cols = []
        for ck in self.col_keys:
            ds_data = self.g_data.get(ck)
            # col = [ds_data[idx] for idx in indices]
            # col = np.array(col, dtype=ds_data.dtype)
            col = h5ds_fast_index(ds_data, indices)
            data_cols.append(col)
        # labels = [self.ds_labels[idx] for idx in indices]
        # labels = np.array(labels, dtype=self.ds_labels.dtype)
        labels = h5ds_fast_index(self.ds_labels, indices)
        return data_cols, labels

    def set_batch_index(self, i):
        self.i = i

    def set_indices(self, indices):
        self.indices = indices
        if self.shuffle: random.shuffle(self.indices)
        self.n_samples = len(self.indices)

        if self.whole_batches:
            self.n_batches = math.floor(self.n_samples / self.batch_size)
            self.n_remain = self.n_samples % self.batch_size
        else:
            self.n_batches = math.ceil(self.n_samples / self.batch_size)
            self.n_remain = 0

        self.i = 0

    def get_split_indices(self, split):
        indices = self.g_splits.get(split)
        indices = list(np.array(indices))
        return indices

    def set_split(self, split):
        self.split = split
        indices = self.g_splits.get(split)
        indices = list(np.array(indices))
        self.set_indices(indices)

    def get_batch_by_index(self, i):
        if i >= self.n_batches:
            raise Exception("batch index exceeds limit")

        start = i * self.batch_size
        end = (i+1) * self.batch_size
        indices = self.indices[start:end]

        return self.get_data_by_indices(indices)

    def get_remaining(self):
        if self.n_remain == 0: return None

        start = self.n_samples - self.n_remain
        end = self.n_samples

        indices = [i for i in range(self.batch_size)]
        indices[:self.n_remain] = self.indices[start:end]

        return self.get_data_by_indices(indices)

    def get_datup_at_index(self, i):
        data_cols = []
        for ck in self.col_keys:
            ds_data = self.g_data.get(ck)
            data_cols.append(ds_data[i])
        return data_cols

    def next_batch(self):
        if self.i < self.n_batches:
            data, labels = self.get_batch_by_index(self.i)
            self.i += 1
            return data, labels
        else:
            return None, None