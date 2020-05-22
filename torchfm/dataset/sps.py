import math
import shutil
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch.utils.data

# Accepts a CSR X. If any of the field_dims are not 1, then
# that column will be interpreted as a feature with multiple
# fields (categorical). Otherwise, everything is considered a
# dense feature and won't be expanded further.
#
# this allows a mixed SVMLight / libFFM format
# where you can have features like 132:2.123 considered numeric
# regular one-hot binary features like 12:1 (also technically numeric)
# but then also features like 100:1, 100:2, 100:3 which are
# categorical with the same field
class SparseDataset(torch.utils.data.Dataset):

    # NN inputs will be len(field_dims) long
    # NNs will have embeddings for indiv fields
    # embedings will be scaled by feature value.
    # the sparse (f64) matrix should hold ints
    # for different field values. a bit messy due to mixed types...
    # but for doubles it's nbd
    def __init__(self, X, y, field_dims, quiet=False):
        self.y = y
        self.X = X
        assert self.X.getformat() == 'csr'
        assert self.X.dtype == np.float64
        if not quiet:
            print('summing dups')
        self.X.sum_duplicates()
        if not quiet:
            print('sorting indices')
        self.X.sort_indices()
        if not quiet:
            print('casting to int')
        self.Xint = self.X.data.astype(np.uint32)
        if not quiet:
            print('casting to float')
        self.X = self.X.astype(np.float32)
        self.field_dims = np.asarray(field_dims, dtype=np.uint32)
        self.with_fields = self.field_dims > 1
        self.num_fields = len(field_dims)
        assert self.X.shape[1] == len(field_dims)

    # returns a two-tuple (field indices, float values, target)
    # each array of length (num fields)
    # float values will usually be 1 for categorical data but may be different
    # for numeric data
    # TODO: may want to make this index-sliceable for faster batching
    def __getitem__(self, index):
        start, stop = self.X.indptr[index], self.X.indptr[index + 1]
        col_idxs = self.X.indices[start:stop]

        fields = np.zeros(self.num_fields, dtype=np.int64)
        values = np.zeros(self.num_fields, dtype=np.float32)

        which_cols_have_fields = self.with_fields[col_idxs]
        fields_for_cols_with_fields = self.Xint[start:stop][which_cols_have_fields]
        cols_with_fields = col_idxs[which_cols_have_fields]
        fields[cols_with_fields] = fields_for_cols_with_fields

        values[col_idxs] = self.X.data[start:stop]
        values[cols_with_fields] = 1

        return fields, values, self.y[index]

    def __len__(self):
        return len(self.y)
