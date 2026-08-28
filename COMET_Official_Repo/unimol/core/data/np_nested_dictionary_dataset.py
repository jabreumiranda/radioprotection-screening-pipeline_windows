# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch
from torch.utils.data.dataloader import default_collate

from . import UnicoreDataset


def _flatten(dico, prefix=None):
    """Flatten a nested dictionary."""
    new_dico = OrderedDict()
    if isinstance(dico, dict):
        prefix = prefix + "." if prefix is not None else ""
        for k, v in dico.items():
            if v is None:
                continue
            new_dico.update(_flatten(v, prefix + k))
    elif isinstance(dico, list):
        for i, v in enumerate(dico):
            new_dico.update(_flatten(v, prefix + ".[" + str(i) + "]"))
    else:
        new_dico = OrderedDict({prefix: dico})
    return new_dico


def _unflatten(dico):
    """Unflatten a flattened dictionary into a nested dictionary."""
    new_dico = OrderedDict()
    for full_k, v in dico.items():
        full_k = full_k.split(".")
        node = new_dico
        for k in full_k[:-1]:
            if k.startswith("[") and k.endswith("]"):
                k = int(k[1:-1])
            if k not in node:
                node[k] = OrderedDict()
            node = node[k]
        node[full_k[-1]] = v
    # print("_unflatten(dico): ", _unflatten(dico))
    return new_dico


class NPNestedDictionaryDataset(UnicoreDataset):
    def __init__(self, defn, mol_dataset=None):
        # print("NPNestedDictionaryDataset __init__ defn: ", defn)
        super().__init__()
        self.defn = _flatten(defn)
        self.mol_dataset = mol_dataset
        first = None
        for v in self.defn.values():
            if not isinstance(
                v,
                (
                    UnicoreDataset,
                    torch.utils.data.Dataset,
                ),
            ):
                raise ValueError("Expected Dataset but found: {}".format(v.__class__))
            first = first or v
            if len(v) > 0:
                assert len(v) == len(first), "dataset lengths must match"

        self._len = len(first)

    def __getitem__(self, index):
        # for k, ds in self.defn.items():
        #     print("NPNestedDictionaryDataset k: ", k )
        #     print("NPNestedDictionaryDataset ds: ", ds)
        #     print("NPNestedDictionaryDataset ds[index]: ", ds[index])
        # print("NPNestedDictionaryDataset __getitem__ self.defn.items(): ",  self.defn.items())
        # for k, ds in self.defn.items():
        #     print("iter k: ", k, "ds: ", ds)
        if self.mol_dataset is None: # behaves like NestedDictionaryDataset
            return OrderedDict((k, ds[index]) for k, ds in self.defn.items())
        else:
            # print("Checkpoint A in NPNestedDictionaryDataset's __getitem__")
            sample = OrderedDict((k, ds[index]) for k, ds in self.defn.items())
            # print("returning sample: ", sample)
            # print("returning sample.keys(): ", sample.keys())
            # if ('target.finetune_target' in sample) and type(sample['target.finetune_target']) is dict:
            #     print("returning sample['target.finetune_target']: ", sample['target.finetune_target'])
            #     print("returning sample['net_input.dataset_name']: ", sample['net_input.dataset_name'])

            return sample
            # TODO: edit this part to pull relevant LNP's data from the child datasets
            # use another dictionary to map LNP index to other ds' index
            # for each NP sample, pull and return all the LNP's components' data: [(src_tokens, src_coord, src_distance, src_edge_type), ..]
            
            # !!TODO: Handle both TTADataset (non-stochastic coord sampling where index runs along the length of (len(self.dataset) or # of mols) * self.conf_size ) and ConformerSampleDataset (stochastic coord sampling where index runs along the length of len(self.dataset) = # of mols )
            

    def __len__(self):
        return self._len

    def collater(self, samples):
        
        # print("nested_dictionary_dataset.py collater, A samples: ", samples)
        # print("nested_dictionary_dataset.py collater, self.defn.items(): ", self.defn.items())
        # odict_items([('net_input.src_tokens', <unimol.core.data.pad_dataset.RightPadDataset object at 0x7f5137cf1ff0>), ('net_input.src_coord', <unimol.data.coord_pad_dataset.RightPadDatasetCoord object at 0x7f5137cf2050>), ('net_input.src_distance', <unimol.core.data.pad_dataset.RightPadDataset2D object at 0x7f5137cf20b0>), ('net_input.src_edge_type', <unimol.core.data.pad_dataset.RightPadDataset2D object at 0x7f5137cf2110>), ('target.finetune_target', <unimol.core.data.raw_dataset.RawLabelDataset object at 0x7f5137cf2170>), ('smi_name', <unimol.core.data.raw_dataset.RawArrayDataset object at 0x7f5137cf21d0>)])

        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        # edit collater to collate all LNP component mol's data into batch data 
        if len(samples) == 0:
            return {}
        sample = OrderedDict()
        # print("collater NPNestedDictionaryDataset samples]: ", samples)
        for k, ds in self.defn.items():
            if k == 'net_input.components':
                # print("A collater NPNestedDictionaryDataset k: ", k)
                # print("A collater NPNestedDictionaryDataset ds: ", ds)
                # print("A collater NPNestedDictionaryDataset [s[k] for s in samples]: ", [s[k] for s in samples])
                sample[k] = [s[k] for s in samples]
                # print("A collater NPNestedDictionaryDataset sample[k]: ", sample[k])
            else:
                try:
                    # print("B collater NPNestedDictionaryDataset k: ", k)
                    # print("B collater NPNestedDictionaryDataset ds: ", ds)
                    # print("B collater NPNestedDictionaryDataset [s[k] for s in samples]: ", [s[k] for s in samples])
                    sample[k] = ds.collater([s[k] for s in samples])
                    # print("collater NPNestedDictionaryDataset sample[k]: ", sample[k])
                except NotImplementedError:
                    sample[k] = default_collate([s[k] for s in samples])
        # print("nested_dictionary_dataset.py collater, B samples: ", samples)
        
        # collect mol data for batch samples from self.mol_dataset
        batch_mol_id2features = OrderedDict()
        # batch_mol_id2features = {}
        mol_id2batch_id = OrderedDict()
        batch_mol_features = []
        batch_mol_ids = []
        
        batch_id = 0

        for s in samples:
            s_components = s['net_input.components']
            for c in s_components:
                c_mol_id = c['mol_id']
                if c_mol_id in batch_mol_id2features: # or in `mol_id2batch_id` or `batch_mol_ids`
                    continue

                # print("before c_features = self.mol_dataset[c_mol_id], c_mol_id: ", c_mol_id, "self.mol_dataset: ", self.mol_dataset)
                c_features = self.mol_dataset[c_mol_id]
                # print("after c_features = self.mol_dataset[c_mol_id], c_mol_id: ", c_mol_id, "self.mol_dataset: ", self.mol_dataset)

                # print("c_mol_id: ", c_mol_id, "c_features smi_name: ", c_features['smi_name'])
                batch_mol_id2features[c_mol_id] = c_features

                mol_id2batch_id[c_mol_id] = batch_id

                batch_mol_features.append(c_features)
                batch_mol_ids.append(c_mol_id)
                batch_id += 1

        # add pad_idx
        mol_id_pad_idx = self.defn['net_input.mol_ids'].pad_idx
        mol_id2batch_id[mol_id_pad_idx] = -1

        mol_features = OrderedDict()
        for k, ds in self.mol_dataset.defn.items():
            try:
                mol_features[k] = ds.collater([s[k] for s in batch_mol_features])
            except NotImplementedError:
                mol_features[k] = default_collate([s[k] for s in batch_mol_features])

        # print("mol_features: ", mol_features)
        # for k, ds in self.defn.items():
        #     if k != "net_input.components":
        #         try:
        #             sample[k] = ds.collater([s[k] for s in samples])
        #         except NotImplementedError:
        #             sample[k] = default_collate([s[k] for s in samples])

        unflattened = _unflatten(sample)
        # print("unflattened keys: ", unflattened.keys())
        unflattened['batch_mol_id2features'] = batch_mol_id2features
        unflattened['mol_id2batch_id'] = mol_id2batch_id
        unflattened['batch_mol_features_list'] = batch_mol_features
        unflattened['batch_mol_ids'] = batch_mol_ids

        unflattened['mol_features'] = mol_features

        mol_batch_ids = unflattened['net_input']['mol_ids']
        # print("mol_id2batch_id A: ", mol_id2batch_id)
        # print("mol_batch_ids A: ", mol_batch_ids)
        mol_batch_ids.apply_(mol_id2batch_id.get) # value that does not correspond to any key gets mapped to pad_idx by default 
        # print("mol_batch_ids B: ", mol_batch_ids)
        # print(batch_mol_features[])
        # print("collater forward, mol_features smi_name: ", mol_features['smi_name'])
        # print("collater forward, input components: ", unflattened['net_input']['components'])
        # print("collater forward mol_batch_ids: ", mol_batch_ids)
        unflattened['mol_batch_ids'] = mol_batch_ids

        # print("unflattened: ", unflattened)

        return unflattened
        # TODO: Return NP input data as well as mol's data here
        # return 1) NP dict detailing a) components' i) mol_id, ii) percent and b) label and 2) original mol batch: _unflatten(sample)
        # return _unflatten(sample)


    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return any(ds.supports_prefetch for ds in self.defn.values())

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        for ds in self.defn.values():
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch(indices)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return all(ds.can_reuse_epoch_itr_across_epochs for ds in self.defn.values())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.defn.values():
            ds.set_epoch(epoch)
