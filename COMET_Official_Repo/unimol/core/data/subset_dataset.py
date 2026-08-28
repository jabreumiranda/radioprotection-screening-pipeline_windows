from . import BaseWrapperDataset
import bisect
import warnings
from typing import (
    Iterable,
)

from torch.utils.data import IterableDataset

# adapted from https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Subset
class SubsetDataset(BaseWrapperDataset):
    def __init__(self, dataset, indices):
        super().__init__(dataset)        
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]
