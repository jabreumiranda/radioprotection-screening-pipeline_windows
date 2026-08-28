# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from functools import lru_cache
import numpy as np

from . import BaseWrapperDataset


class NoiseAugmentDataset(BaseWrapperDataset):
    def __init__(self, dataset, noise=0.01, noise_type="normal_proportionate", trunc_value=2):
        super().__init__(dataset)
        self.noise = noise # stdev for normal, max for uniform
        self.noise_type = noise_type
        self.trunc_value = trunc_value

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        # torch.randn_like returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1
        if self.noise_type == "trunc_normal_proportionate":
            noise = torch.clamp(torch.randn_like(self.dataset[idx]), min=-self.trunc_value, max=self.trunc_value) * (self.dataset[idx] * self.noise)
            return self.dataset[idx] + noise
        
        elif self.noise_type == "normal_proportionate":
            noise = torch.randn_like(self.dataset[idx]) * (self.dataset[idx] * self.noise)
            # print("noise: ", noise)
            # print("self.dataset[idx]: ", self.dataset[idx])
            return self.dataset[idx] + noise
        
        # torch.rand_like returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0, 1)
        elif self.noise_type == "uniform_proportionate":
            uniform_noise = -1 + 2*torch.rand_like(self.dataset[idx]) # uniform noise in [-1, 1)
            noise = uniform_noise * (self.dataset[idx] * self.noise)
            return self.dataset[idx] + noise
        else:
            return self.dataset[idx]


