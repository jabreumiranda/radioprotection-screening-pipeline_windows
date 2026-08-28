# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from unimol.core.data import BaseWrapperDataset


class KeyDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key
        # print("KeyDataset self.key: ", self.key)

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        # print("KeyDataset idx: ", idx, "self.dataset: ", self.dataset)
        # print("__getitem__, self.key: ", self.key)
        # if self.key == 'reaction_step':
        #     print("reaction_step __getitem__: ", self.dataset[idx][self.key])
        # print("type(self.dataset[idx]): ", type(self.dataset[idx]))
        return self.dataset[idx][self.key]

class KeyDatasetWithDefaultValue(BaseWrapperDataset):
    def __init__(self, dataset, key, default_value='default'):
        self.dataset = dataset
        self.key = key
        self.default_value = default_value
        # print("KeyDataset self.key: ", self.key)

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        # print("KeyDatasetWithDefaultValue idx: ", idx, "self.dataset[idx]: ", self.dataset[idx])
        # print("__getitem__, self.key: ", self.key)
        # if self.key == 'NP_ratio':
        #     print("NP_ratio __getitem__: ", self.dataset[idx].get(self.key, self.default_value))
        return self.dataset[idx].get(self.key, self.default_value)

# dataset to indicate whether sample has key (e.g. np_props) or not
class KeyExistDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.key in self.dataset[idx]:
            return True
        else:
            return False