# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data.dataloader import default_collate
from functools import lru_cache
from . import UnicoreDataset


class RawLabelDataset(UnicoreDataset):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels
    
    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        sample = self.labels[index]
        # print("sample: ", sample)
        return sample

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        # print("samples: ", samples)
        collated_samples =  torch.tensor(samples)
        # print("collated_samples: ", collated_samples)
        return collated_samples

# TODO NOW: add logic to check for dataset name to retrieve the respective label, to handle optional label in collater fn
class MultiDatasetDictLabelDataset(UnicoreDataset):
    def __init__(self, labels, task_schema=None, label_pad=0):
        super().__init__()
        self.labels = labels
        self.task_schema = task_schema
        self.label_pad = label_pad
    
    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        sample = self.labels[index]
        # print("sample: ", sample) # dict: keys are task names, values are labels
        return sample

    def __len__(self):
        return len(self.labels)
    
    # TODO NOW: make new MultiDatasetDictLabelDataset to handle optional label in collater fn
    # TODO NOW: Need to add mechanism to check if sample is using optional label for loss compute
    # collater will include a loss_mask to indicate which labels should be left out of loss computation: 1 for loss, 0 for no loss
    def collater(self, samples): # `samples` is a list of dicts
        # print("collater samples: ", samples)
        label_dict = {}
        for task_name in self.task_schema:
            mask_name = task_name + "_mask"
            mask = []
            task_samples = []
            for s in samples:
                if task_name in s:
                    task_samples.append(s[task_name])
                    mask.append(True)
                else:
                    task_samples.append(self.label_pad)
                    mask.append(False)
                    
            # task_samples = [s[task_name] for s in samples]

            label_dict[task_name] = torch.tensor(task_samples)
            label_dict[mask_name] = torch.tensor(mask)
        # collated_samples =  torch.tensor(samples)
        # print("label_dict: ", label_dict)
        return label_dict
    
class DictLabelDataset(UnicoreDataset):
    def __init__(self, labels, task_schema=None):
        super().__init__()
        self.labels = labels
        self.task_schema = task_schema
    
    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        sample = self.labels[index]
        # print("sample: ", sample) # dict: keys are task names, values are labels
        return sample

    def __len__(self):
        return len(self.labels)

    def collater(self, samples): # `samples` is a list of dicts
        label_dict = {}
        for task_name in self.task_schema:
            task_samples = [s[task_name] for s in samples]
            label_dict[task_name] = torch.tensor(task_samples)
        # collated_samples =  torch.tensor(samples)
        print("collater, label_dict: ", label_dict)
        return label_dict
    
class RawArrayDataset(UnicoreDataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)


class RawNumpyDataset(UnicoreDataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return torch.from_numpy(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)
