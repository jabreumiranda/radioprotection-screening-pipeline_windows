# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unimol.core.data import (
    Dictionary,
    NestedDictionaryDataset,
    NPNestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    SubsetDataset,
    # ConcatDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
    MultiDatasetDictLabelDataset,
    DictLabelDataset,
    RawArrayDataset,
    FromNumpyDataset,
    NoiseAugmentDataset
)
from unimol.data import (
    KeyDataset,
    KeyDatasetWithDefaultValue,
    KeyExistDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    data_utils,
)
# import unimol
# print("unimol.__file__: ", unimol.__file__)
# import unimol.data
# print("unimol.data.__file__: ", unimol.data.__file__)
# import unimol.core
# print("unimol.core.__file__: ", unimol.core.__file__)
# from unimol.core import (
#     distributed,
# )
# print("distributed.__file__: ", distributed.__file__)

from unimol.data.tta_dataset import TTADataset
from unimol.core.tasks import UnicoreTask, register_task
from unimol.core.logging import meters, metrics
import json
from torch.utils.data import ConcatDataset

logger = logging.getLogger(__name__)

task_metainfo = {
    "esol": {
        "mean": -3.0501019503546094,
        "std": 2.096441210089345,
        "target_name": "logSolubility",
    },
    "freesolv": {
        "mean": -3.8030062305295944,
        "std": 3.8478201171088138,
        "target_name": "freesolv",
    },
    "lipo": {"mean": 2.186336, "std": 1.203004, "target_name": "lipo"},
    "qm7dft": {
        "mean": -1544.8360893118609,
        "std": 222.8902092792289,
        "target_name": "u0_atom",
    },
    "qm8dft": {
        "mean": [
            0.22008500524052105,
            0.24892658759891675,
            0.02289283121913152,
            0.043164444107224746,
            0.21669716560818883,
            0.24225989336408812,
            0.020287111373358993,
            0.03312609817084387,
            0.21681478862847584,
            0.24463634931699113,
            0.02345177178004201,
            0.03730141834205415,
        ],
        "std": [
            0.043832862248693226,
            0.03452326954549232,
            0.053401140662012285,
            0.0730556474716259,
            0.04788020599385645,
            0.040309670766319,
            0.05117163534626215,
            0.06030064428723054,
            0.04458294838213221,
            0.03597696243350195,
            0.05786865052149905,
            0.06692733477994665,
        ],
        "target_name": [
            "E1-CC2",
            "E2-CC2",
            "f1-CC2",
            "f2-CC2",
            "E1-PBE0",
            "E2-PBE0",
            "f1-PBE0",
            "f2-PBE0",
            "E1-CAM",
            "E2-CAM",
            "f1-CAM",
            "f2-CAM",
        ],
    },
    "qm9dft": {
        "mean": [-0.23997669940621352, 0.011123767412331285, 0.2511003712141015],
        "std": [0.02213143402267657, 0.046936069870866196, 0.04751888787058615],
        "target_name": ["homo", "lumo", "gap"],
    },
}


@register_task("mol_np_finetune")
class UniMolNPFinetuneTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen ",
        )

        # arg for interpretation and explanations
        parser.add_argument(
            "--explanation-save-path",
            default="explanations/explanations.pt",
            type=str,
            help="path to save explanations",
        )
        parser.add_argument(
            "--input-explanation-schema",
            default="input_explanation_schema.json",
            type=str,
            help="path to explanation schema",
        )
        parser.add_argument(
            "--explanation-save-interval",
            default=-1,
            type=int,
            help="sample interval to save explanations",
        )

        # for concatenating datasets
        parser.add_argument(
            "--concat-datasets",
            action="store_true",
            help="concatenate datasets for training",
        )

        parser.add_argument(
            '--epoch-to-stop', 
            default=None, 
            type=int, 
            metavar='N',
            help='stop training at specified epoch to replicate early stopping during valid step, to prevent overfitting without changing the max epoch (affects lr schedule)')
        
        parser.add_argument("--output-cls-rep", action="store_true", help="whether to output cls representation during inference")

        # SSLNP todo: add args for SSLNP COMET
        parser.add_argument(
            "--include-excipients",
            action="store_true",
            help="include excipients in inference and the dataset",
        )



        # parser.add_argument(
        #     "--explanation-save-path",
        #     default="explanations.pt",
        #     type=str,
        #     help="path to save explanations",
        # )

    def __init__(self, args, dictionary, component_type_dictionary=None):
        
        print("unimol_np_finetune __init__")

        super().__init__(args)
        self.task_schema = None
        self.np_prop_schema = None
        self.dictionary = dictionary # atom dictionary

        def list_dirs(path):
            return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        self.subdataset_names = list_dirs(os.path.join(self.args.data, self.args.task_name))
        
        # dictionary for np components
        self.component_type_dictionary = component_type_dictionary

        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif self.args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True
        if self.args.task_name in task_metainfo:
            # for regression task, pre-compute mean and std
            self.mean = task_metainfo[self.args.task_name]["mean"]
            self.std = task_metainfo[self.args.task_name]["std"]
        else:
            self.mean = None
            self.std = None

    @classmethod
    def setup_task_backup(cls, args, **kwargs):
        
        print("unimol_np_finetune setup_task") # used by unimol_np_finetune.py and unimol's infer.py

        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        component_type_dictionary = Dictionary.load(os.path.join(args.data, 'component_type_dict.txt'))

        logger.info("dictionary: {} types".format(len(dictionary)))
        # print("dictionary component_type_dictionary type: ", type(component_type_dictionary))
        return cls(args, dictionary, component_type_dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        

        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        
        if args.full_dataset_task_schema_path is not None and '.json' in args.full_dataset_task_schema_path:
            component_type_dictionaries = Dictionary.load_component_types_from_master_schema_json(os.path.join(args.data, args.full_dataset_task_schema_path))
            logger.info("json dictionaries: {} types".format(len(component_type_dictionaries)))
            return cls(args, dictionary, component_type_dictionaries)
        else:
            if args.component_types_schema_path is not None and '.json' in args.component_types_schema_path:
                component_type_dictionaries = Dictionary.load_from_json(os.path.join(args.data, args.component_types_schema_path))
                logger.info("json dictionaries: {} types".format(len(component_type_dictionaries)))
                
                return cls(args, dictionary, component_type_dictionaries)
            else:
                component_type_dictionary = Dictionary.load(os.path.join(args.data, args.component_types_schema_path))
                logger.info("dictionary: {} types".format(len(dictionary)))

                return cls(args, dictionary, component_type_dictionary)


    def load_concat_dataset(self, split, dropped_datasets=None, **kwargs):
        # , subdatasets_to_load=None
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """

        """
        [
        {'components': 
            [
                {
                'atoms': ['O', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'N', 'C', 'N', 'C', 'N', 'N', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], 
                'coordinates': [array([[-1.4622144 , -1.3234322 ,  2.5572178 ],
                'mol': <rdkit.Chem.rdchem.Mol object at 0x7f3c7c115f50>, 'smi': 'O1c2c(cc(cc2)-c2cc(ccc2)C#N)[C@]2(N=C(N)N(C)C2=O)CC1(C)C', 'name': 'IL-1', 'percent': 100
                }, ..
            ], 
        'target': 1}
        ]
        """

        mol_path = os.path.join(self.args.data, self.args.task_name, "mol.lmdb")

        if dropped_datasets == None or len(dropped_datasets) == 0:
            subdataset_names_to_load = self.subdataset_names
        else:
            subdataset_names_to_load = [subdataset_name for subdataset_name in self.subdataset_names if subdataset_name not in dropped_datasets]

        if hasattr(self.args, "leave_last_subdataset_to_train") and self.args.leave_last_subdataset_to_train and len(subdataset_names_to_load) == 0:
            print("self.args.leave_last_subdataset_to_train and len(subdataset_names_to_load) == 0")
            subdataset_names_to_load = self.prev_subdataset_names_to_load
        print("dropped_datasets: ", dropped_datasets)
        print("subdataset_names_to_load: ", subdataset_names_to_load)
        self.prev_dropped_datasets = dropped_datasets
        self.prev_subdataset_names_to_load = subdataset_names_to_load

        subdataset_split_paths = [os.path.join(self.args.data, self.args.task_name, subdataset, split + ".lmdb") for subdataset in subdataset_names_to_load]
        dataset = LMDBDataset(mol_path) # contains detailed mol's data such as coords
        
        # indexed dataset version
        """
        Create lmdb dataset with:
        a) NP compositions: list of components where each component contain molecule_id
        b) molecule dataset: like unimol original dataset
        """

        smi_dataset = KeyDataset(dataset, "smi")
        # random sample one out of conf_size conformations
        sample_dataset = ConformerSampleDataset(
            dataset, self.args.seed, "atoms", "coordinates"
        )
        dataset = AtomTypeDataset(dataset, sample_dataset)

        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        )
        dataset = CroppingDataset(
            dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
        )
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        # add [CLS] token at the front
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))

        coord_dataset = FromNumpyDataset(coord_dataset)
        
        # add [CLS] token's coord (0,0) at the front
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)
        
        # Create nested dataset of all molecules' information
        mol_nest_dataset = NPNestedDictionaryDataset(
            {
                # "net_input": { # to rename as mol_data
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": RightPadDatasetCoord(
                    coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": RightPadDataset2D(
                    distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
                "smi_name": RawArrayDataset(smi_dataset),
            },
        )
        
        np_subdataset_list_to_concat = []
        for split_path in subdataset_split_paths:
            np_dataset = LMDBDataset(split_path) # contains NP's label and its components' reference (mol_id), name and percent
            np_subdataset_list_to_concat.append(np_dataset)
        
        # Concatenate multiple subdatasets
        concat_np_dataset = ConcatDataset(np_subdataset_list_to_concat)

        if self.args.train_data_ratio < 1.0 and split == "train":
            train_data_size = int(len(concat_np_dataset) * self.args.train_data_ratio)
            with data_utils.numpy_seed(self.args.seed):
                subset_indices = np.random.choice(range(len(concat_np_dataset)), size=train_data_size, replace=False)

            subset_np_dataset = SubsetDataset(
                dataset=concat_np_dataset,
                indices=subset_indices,
            )

            concat_np_dataset = subset_np_dataset
            print("concat_np_dataset Checkpoint B: ", len(concat_np_dataset))

        # Set up NP dataset
        mol_id_dataset = FromNumpyDataset(KeyDataset(concat_np_dataset, "mol_id"), convert_to_np=True)
        percent_dataset = FromNumpyDataset(KeyDataset(concat_np_dataset, "percent"), convert_to_np=True)

        # noise augmentation for percent_dataset
        if self.args.noise_augment_percent and split == "train":
            percent_dataset = NoiseAugmentDataset(percent_dataset, noise=self.args.percent_noise, noise_type=self.args.percent_noise_type)

        # SSLNP todo: add composition_enc_types data for SSLNP COMET
        excipient_datasets = {}
        if self.args.include_excipients:
            composition_enc_types_dataset = FromNumpyDataset(KeyDataset(concat_np_dataset, "composition_enc_types"), convert_to_np=True)
            composition_enc_types_dataset = RightPadDataset(
                                            composition_enc_types_dataset,
                                            pad_idx=0,
                                            pad_to_multiple=2,
                                        ) # mirror format of percent_dataset
            excipient_datasets["composition_enc_types"] = composition_enc_types_dataset


        # Add custom component_type_dataset here - start
        # look for str 'component_type' in key to build <component_type> dataset here
        if type(self.component_type_dictionary) == dict:
            component_type_datasets = {}

            # Make component_type_dataset
            if 'component_type' in self.component_type_dictionary:
                
                component_type_dataset = KeyDataset(concat_np_dataset, "component_type")
                component_type_dataset = TokenizeDataset(
                    component_type_dataset, self.component_type_dictionary['component_type']['dictionary']
                )
                component_type_datasets["component_type"] = component_type_dataset

            # Make component_type subclass dataset: e.g. reaction_step
            for key_value in self.component_type_dictionary:
                if key_value != 'component_type':
                    key_value_in_lmdb = key_value
                    component_type_subclass_dataset = KeyDataset(concat_np_dataset, key_value_in_lmdb)
                    component_type_subclass_dataset = TokenizeDataset(
                        component_type_subclass_dataset, self.component_type_dictionary[key_value_in_lmdb]['dictionary']
                    )
                    component_type_datasets[key_value] = component_type_subclass_dataset
            
        else:
            component_type_dataset = KeyDataset(concat_np_dataset, "component_type")
            component_type_dataset = TokenizeDataset(
                component_type_dataset, self.component_type_dictionary
            )

        tgt_dataset = KeyDataset(concat_np_dataset, "target")
        components_dataset = KeyDataset(concat_np_dataset, "components")
        dataset_name_dataset = KeyDatasetWithDefaultValue(concat_np_dataset, "dataset_name", "default")
        
        # to record preds with lnp_id
        lnp_id_dataset = KeyDatasetWithDefaultValue(concat_np_dataset, "lnp_id", default_value="[PAD]") # fill missing np_prop value with [PAD]

        # make lnp-wide prop dataset
        if (self.args.full_dataset_task_schema_path is not None and '.json' in self.args.full_dataset_task_schema_path) or (self.args.np_prop_schema_path is not None and '.json' in self.args.np_prop_schema_path):
            prop_dataset = {}
            prop_mask_dataset = {}
            for prop in self.np_prop_schema:
                if 'type' in self.np_prop_schema[prop] and self.np_prop_schema[prop]['type'] == 'categorical':
                    # handle categorical prop here
                    cur_prop_dataset = KeyDatasetWithDefaultValue(concat_np_dataset, prop, default_value="[PAD]") # fill missing np_prop value with [PAD]

                    cur_prop_dictionary = Dictionary.load_from_list(self.np_prop_schema[prop]['dictionary'])
                    prop_dataset[prop] = TokenizeDataset(
                        cur_prop_dataset, cur_prop_dictionary
                    )
                    prop_mask_name = prop + "_mask"
                    # to indicate whether sample has key (e.g. np_props) or not
                    prop_mask_dataset[prop_mask_name] = KeyExistDataset(concat_np_dataset, prop)
                    
                else: # continuous
                    prop_dataset[prop] = KeyDatasetWithDefaultValue(concat_np_dataset, prop, default_value=0) # fill missing np_prop value with 0
                    prop_mask_name = prop + "_mask"
                    # to indicate whether sample has key (e.g. np_props) or not
                    prop_mask_dataset[prop_mask_name] = KeyExistDataset(concat_np_dataset, prop)


        # Set up sub dictionary dataset for component_type* datasets
        nest_component_type_datasets = {}
        for key_value in component_type_datasets:
            input_key_value = key_value + "s" # pluralize input names
            nest_component_type_datasets[input_key_value] = RightPadDataset(
                        component_type_datasets[key_value],
                        pad_idx=self.component_type_dictionary[key_value]['dictionary'].pad() if (type(self.component_type_dictionary) == dict) else self.component_type_dictionary.pad(),
                        pad_to_multiple=2,
                        # pad_to_multiple=1,
                    )
        # make target dataset
        
        if self.args.full_dataset_task_schema_path is not None and '.json' in self.args.full_dataset_task_schema_path:
            label_dataset = MultiDatasetDictLabelDataset(tgt_dataset, self.task_schema)
        elif self.args.tasks_schema_path is not None and '.json' in self.args.tasks_schema_path:
            label_dataset = DictLabelDataset(tgt_dataset, self.task_schema)
        else:
            label_dataset = RawLabelDataset(tgt_dataset)

        if self.np_prop_schema is not None:
            for prop in self.np_prop_schema:
                prop_dataset[prop] = RawLabelDataset(prop_dataset[prop])
                prop_mask_name = prop + "_mask"
                prop_mask_dataset[prop_mask_name] = RawLabelDataset(prop_mask_dataset[prop_mask_name])
        else:
            prop_dataset = {}
            prop_mask_dataset = {}

        # build nested dataset to nest multiple dataset sources, >1 datasets in the NPNestedDictionaryDataset object
        nest_dataset = NPNestedDictionaryDataset(
            {                
                "target": {
                    "finetune_target": label_dataset,
                },

                # component dataset
                "net_input": {
                    # datasets for 'mol_id', 'percent', 'component_type'
                    "mol_ids": RightPadDataset(
                        mol_id_dataset,
                        pad_idx=len(dataset), # set pad_idx as the (last mol_id + 1)
                        pad_to_multiple=2,
                        # pad_to_multiple=1,
                    ),
                    "percents": RightPadDataset(
                        percent_dataset,
                        pad_idx=0,
                        pad_to_multiple=2,
                        # pad_to_multiple=1,
                    ),
                    "lnp_ids": RawArrayDataset(lnp_id_dataset),
                    
                    # set custom <component_type> dataset here
                    **nest_component_type_datasets,
                    **prop_dataset,
                    **prop_mask_dataset,
                    **excipient_datasets,


                    "components": RawArrayDataset(components_dataset),
                    "dataset_name": RawArrayDataset(dataset_name_dataset),
                }
                # "smi_name": RawArrayDataset(smi_dataset),
            },
            mol_dataset=mol_nest_dataset
        )

        print("len(components_dataset): ", len(components_dataset))
        print("len(nest_dataset): ", len(nest_dataset))

        # Shuffle dataset
        if not self.args.no_shuffle and split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(nest_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """

        """
        [
        {'components': 
            [
                {
                'atoms': ['O', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'N', 'C', 'N', 'C', 'N', 'N', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], 
                'coordinates': [array([[-1.4622144 , -1.3234322 ,  2.5572178 ],
                'mol': <rdkit.Chem.rdchem.Mol object at 0x7f3c7c115f50>, 'smi': 'O1c2c(cc(cc2)-c2cc(ccc2)C#N)[C@]2(N=C(N)N(C)C2=O)CC1(C)C', 'name': 'IL-1', 'percent': 100
                }, ..
            ], 
        'target': 1}
        ]
        """
        mol_path = os.path.join(self.args.data, self.args.task_name, "mol.lmdb")
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")

        dataset = LMDBDataset(mol_path) # contains detailed mol's data such as coords
        np_dataset = LMDBDataset(split_path) # contains NP's label and its components' reference (mol_id), name and percent 
        # get subset of train dataset
        if self.args.train_data_ratio < 1.0 and split == "train":
            train_data_size = int(len(np_dataset) * self.args.train_data_ratio)
            with data_utils.numpy_seed(self.args.seed):
                subset_indices = np.random.choice(range(len(np_dataset)), size=train_data_size, replace=False)

            subset_np_dataset = SubsetDataset(
                dataset=np_dataset,
                indices=subset_indices,
            )

            np_dataset = subset_np_dataset


        # indexed dataset version
        """
        Create lmdb dataset with:
        a) NP compositions: list of components where each component contain molecule_id
        b) molecule dataset: like unimol original dataset
        """

        # Set up mol dataset
        smi_dataset = KeyDataset(dataset, "smi")
        # random sample one out of conf_size conformations
        sample_dataset = ConformerSampleDataset(
            dataset, self.args.seed, "atoms", "coordinates"
        )
        dataset = AtomTypeDataset(dataset, sample_dataset)

        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        )
        dataset = CroppingDataset(
            dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
        )
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        # add [CLS] token at the front
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))

        coord_dataset = FromNumpyDataset(coord_dataset)
        
        # add [CLS] token's coord (0,0) at the front
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)
        
        mol_id_dataset = FromNumpyDataset(KeyDataset(np_dataset, "mol_id"), convert_to_np=True)
        percent_dataset = FromNumpyDataset(KeyDataset(np_dataset, "percent"), convert_to_np=True)

        # noise augmentation for percent_dataset
        if self.args.noise_augment_percent and split == "train":
            percent_dataset = NoiseAugmentDataset(percent_dataset, noise=self.args.percent_noise, noise_type=self.args.percent_noise_type)

        # SSLNP todo: add composition_enc_types data for SSLNP COMET
        excipient_datasets = {}
        if self.args.include_excipients:
            composition_enc_types_dataset = FromNumpyDataset(KeyDataset(np_dataset, "composition_enc_types"), convert_to_np=True)
            excipient_datasets["composition_enc_types"] = composition_enc_types_dataset

        # Add custom component_type_dataset here - start
        # look for str 'component_type' in key to build <component_type> dataset here
        if type(self.component_type_dictionary) == dict:
            component_type_datasets = {}

            # Make component_type_dataset
            if 'component_type' in self.component_type_dictionary:                
                component_type_dataset = KeyDataset(np_dataset, "component_type")
                component_type_dataset = TokenizeDataset(
                    component_type_dataset, self.component_type_dictionary['component_type']['dictionary']
                )
                component_type_datasets["component_type"] = component_type_dataset

            # Make component_type subclass dataset: e.g. reaction_step
            for key_value in self.component_type_dictionary:
                if key_value != 'component_type':
                    key_value_in_lmdb = key_value
                    component_type_subclass_dataset = KeyDataset(np_dataset, key_value_in_lmdb)
                    component_type_subclass_dataset = TokenizeDataset(
                        component_type_subclass_dataset, self.component_type_dictionary[key_value_in_lmdb]['dictionary']
                    )
                    component_type_datasets[key_value] = component_type_subclass_dataset
            
        else:
            # print("self.component_type_dictionary B: ", self.component_type_dictionary)
            component_type_dataset = KeyDataset(np_dataset, "component_type")
            component_type_dataset = TokenizeDataset(
                component_type_dataset, self.component_type_dictionary
            )
        # Add custom component_type_dataset here - end

        tgt_dataset = KeyDataset(np_dataset, "target")
        components_dataset = KeyDataset(np_dataset, "components")
        dataset_name_dataset = KeyDatasetWithDefaultValue(np_dataset, "dataset_name", "default")

        # make lnp-wide prop dataset
        if (self.args.full_dataset_task_schema_path is not None and '.json' in self.args.full_dataset_task_schema_path) or (self.args.np_prop_schema_path is not None and '.json' in self.args.np_prop_schema_path):
        # if self.np_prop_schema is not None:
            prop_dataset = {}
            prop_mask_dataset = {}
            for prop in self.np_prop_schema:
                prop_dataset[prop] = KeyDatasetWithDefaultValue(np_dataset, prop, default_value=0) # fill missing np_prop value with 0
                prop_mask_name = prop + "_mask"
                # to indicate whether sample has key (e.g. np_props) or not
                prop_mask_dataset[prop_mask_name] = KeyExistDataset(np_dataset, prop)

        # Create nested dataset of all information
        mol_nest_dataset = NPNestedDictionaryDataset(
            {
                # "net_input": { # to rename as mol_data
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": RightPadDatasetCoord(
                    coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": RightPadDataset2D(
                    distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
                "smi_name": RawArrayDataset(smi_dataset),
            },
        )

        # Set up sub dictionary dataset for component_type* datasets
        nest_component_type_datasets = {}
        for key_value in component_type_datasets:
            input_key_value = key_value + "s" # pluralize input names
            nest_component_type_datasets[input_key_value] = RightPadDataset(
                        component_type_datasets[key_value],
                        pad_idx=self.component_type_dictionary[key_value]['dictionary'].pad() if (type(self.component_type_dictionary) == dict) else self.component_type_dictionary.pad(),
                        pad_to_multiple=2,
                        # pad_to_multiple=1,
                    )

        if self.args.full_dataset_task_schema_path is not None and '.json' in self.args.full_dataset_task_schema_path:
            # print("load_dataset self.task_schema: ", self.task_schema)
            label_dataset = MultiDatasetDictLabelDataset(tgt_dataset, self.task_schema)
        elif self.args.tasks_schema_path is not None and '.json' in self.args.tasks_schema_path:
            # print("load_dataset self.task_schema: ", self.task_schema)
            label_dataset = DictLabelDataset(tgt_dataset, self.task_schema)
        else:
            label_dataset = RawLabelDataset(tgt_dataset)

        if self.np_prop_schema is not None:
            for prop in self.np_prop_schema:
                prop_dataset[prop] = RawLabelDataset(prop_dataset[prop])
                prop_mask_name = prop + "_mask"
                prop_mask_dataset[prop_mask_name] = RawLabelDataset(prop_mask_dataset[prop_mask_name])
        else:
            prop_dataset = {}
            prop_mask_dataset = {}

        # build nested dataset to nest multiple dataset sources, >1 datasets in the NPNestedDictionaryDataset object
        nest_dataset = NPNestedDictionaryDataset(
            {                
                "target": {
                    "finetune_target": label_dataset,
                },

                # component dataset
                "net_input": {
                    # datasets for 'mol_id', 'percent', 'component_type'
                    "mol_ids": RightPadDataset(
                        mol_id_dataset,
                        pad_idx=len(dataset), # set pad_idx as the (last mol_id + 1)
                        # pad_idx=self.dictionary.pad(),
                        pad_to_multiple=2,
                        # pad_to_multiple=1,
                    ),
                    "percents": RightPadDataset(
                        percent_dataset,
                        pad_idx=0,
                        # pad_idx=self.dictionary.pad(),
                        pad_to_multiple=2,
                        # pad_to_multiple=1,
                    ),

                    # set custom <component_type> dataset here
                    **nest_component_type_datasets,
                    **prop_dataset,
                    **prop_mask_dataset,
                    **excipient_datasets,

                    "components": RawArrayDataset(components_dataset),
                    "dataset_name": RawArrayDataset(dataset_name_dataset),
                }
                # "smi_name": RawArrayDataset(smi_dataset),
            },
            mol_dataset=mol_nest_dataset
        )

        if not self.args.no_shuffle and split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(components_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unimol.core import models
        

        model = models.build_model(args, self)

        label_schema_present = False
        # Multiple datasets, multiple tasks
        if args.full_dataset_task_schema_path is not None and '.json' in args.full_dataset_task_schema_path:
            full_labels_dict = {}
            with open(os.path.join(args.data, args.full_dataset_task_schema_path), 'r') as openfile:
                # Reading from json file
                master_schema = json.load(openfile)
                for dataset_name in master_schema["datasets"]:
                    dataset_labels_dict = master_schema["datasets"][dataset_name]["labels"]

                    for label_name in dataset_labels_dict:
                        new_label_name = label_name

                        if new_label_name in full_labels_dict and dataset_labels_dict[label_name] != full_labels_dict[new_label_name]:
                            raise RuntimeError("repeated label_name {} with different values: {}, {}".format(new_label_name, dataset_labels_dict[label_name], full_labels_dict[new_label_name]))

                        # new_label_name = dataset_name + "|" + label_name
                        full_labels_dict[new_label_name] = dataset_labels_dict[label_name]

            self.task_schema = full_labels_dict
            label_schema_present = True
        
        # Single dataset, multiple task
        elif args.tasks_schema_path is not None and '.json' in args.tasks_schema_path:
            with open(os.path.join(args.data, args.tasks_schema_path), 'r') as openfile:
                # Reading from json file
                task_schema = json.load(openfile)
                self.task_schema = task_schema
            label_schema_present = True

        if label_schema_present:
            if self.args.cls_head_config == 'multi':
                for task_name in self.task_schema:
                    model.register_classification_head(
                        task_name,
                        num_classes=self.args.num_classes,
                        # inner_dim=self.args.lnp_encoder_embed_dim,
                    )
            elif self.args.cls_head_config == 'single': # only one classification head
                model.register_classification_head(
                    self.args.classification_head_name,
                    num_classes=self.args.num_classes,
                )
        else:
            self.task_schema = None
            # if schema is not used for multitask training, then use classification head named `classification_head_name`
            model.register_classification_head(
                self.args.classification_head_name,
                num_classes=self.args.num_classes,
                # inner_dim=self.args.lnp_encoder_embed_dim,
            )
        
        # get np_props from schema
        if self.args.full_dataset_task_schema_path is not None and '.json' in self.args.full_dataset_task_schema_path:
            np_prop_schema = {}
            with open(os.path.join(args.data, args.full_dataset_task_schema_path), 'r') as openfile:
                # Reading from json file
                master_schema = json.load(openfile)
                for dataset_name in master_schema["datasets"]:
                    dataset_schema = master_schema["datasets"][dataset_name]
                    if "np_props" in dataset_schema:
                        np_prop_schema = {**np_prop_schema, **dataset_schema["np_props"]} 

            self.np_prop_schema = np_prop_schema
        else:
            if self.args.np_prop_schema_path is not None and '.json' in self.args.np_prop_schema_path:
                with open(os.path.join(args.data, args.np_prop_schema_path), 'r') as openfile:
                    # Reading from json file
                    np_prop_schema = json.load(openfile)
                    self.np_prop_schema = np_prop_schema
            else:
                self.np_prop_schema = None


        return model
