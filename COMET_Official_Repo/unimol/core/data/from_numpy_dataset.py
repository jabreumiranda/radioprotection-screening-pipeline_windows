# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from functools import lru_cache
import numpy as np

from . import BaseWrapperDataset


class FromNumpyDataset(BaseWrapperDataset):
    def __init__(self, dataset, convert_to_np=False):
        super().__init__(dataset)
        self.convert_to_np = convert_to_np

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.convert_to_np:
            return torch.from_numpy(np.asarray(self.dataset[idx]))
        else:
            return torch.from_numpy(self.dataset[idx])


