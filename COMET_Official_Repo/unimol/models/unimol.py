# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unimol.core import utils
from unimol.core.data import data_utils
from unimol.core.models import BaseUnicoreModel, register_model, register_model_architecture
from unimol.core.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from .lnp_transformer_encoder import LNPTransformerEncoder
from typing import Dict, Any, List
import os
import json
import argparse


logger = logging.getLogger(__name__)


@register_model("np_unimol")
class NPUniMolModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add LNP model-specific arguments to the parser."""

        def float_range(min_value, max_value):
            def check_float(value):
                float_value = float(value)
                if float_value < min_value or float_value > max_value:
                    raise argparse.ArgumentTypeError(f"Value must be between {min_value} and {max_value}")
                return float_value
            return check_float
        
        parser.add_argument(
            "--lnp-encoder-layers", type=int, metavar="L", help="num encoder layers for LNP model"
        )
        parser.add_argument(
            "--lnp-encoder-embed-dim",
            type=int,
            metavar="H",
            help="LNP encoder embedding dimension",
        )
        parser.add_argument(
            "--lnp-encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="LNP encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--lnp-encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads for LNP model",
        )
        parser.add_argument('--load-full-np-model', default=False, action='store_true',
                        help='load weights of full LNP model or only those of its mol_model')

        # args for cls token and head
        parser.add_argument(
            "--cls-emb-config",
            choices=['single', 'multi'],
            # default="single",
            default="multi",
            help="whether to use single or multi cls token for multi-task learning",
        )
        parser.add_argument(
            "--cls-head-config",
            choices=['single', 'multi'],
            # default="single",
            default="multi",
            help="whether to use single or multi reg heads for multi-task learning",
        )

        """
        LNP model-specific args to add: <arg value> (current placeholder value)
        percent_embed_dim (K)
        component_types_embed_dim (encoder_embed_dim)
        """
        parser.add_argument(
            "--percent-embed-dim",
            type=int,
            metavar="F",
            help="Percent value embedding dimension",
        )
        parser.add_argument(
            "--component-types-embed-dim",
            type=int,
            metavar="F",
            help="LNP component type embedding dimension",
        )


        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--loss-sample-dropout", type=float, metavar="D", help="loss sample dropout probability"
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )

        # Custom arg for NP task
        parser.add_argument(
            "--full-dataset-task-schema-path",
            # default=None,
            default="pdna_only_master_schema.json",
            help="dictionary file for multiple datasets, key is dataset name, value is a dictionary containing tasks_schema_path, component_types_schema_path and np_prop_schema_path as keys",
        )
        parser.add_argument(
            "--component-types-schema-path",
            # default="dummy_lnp_component_type_schema.json",
            default="pbae_component_type_schema.json",
            help="to deprecate and replace with 'full_dataset_task_schema_path': dictionary file for component_types",
        )
        parser.add_argument(
            "--tasks-schema-path",
            # default=None,
            default="pbae_task_schema.json",
            help="to deprecate and replace with 'full_dataset_task_schema_path': dictionary file for task's label",
        )
        parser.add_argument(
            "--np-prop-schema-path",
            default=None,
            help="to deprecate and replace with 'full_dataset_task_schema_path': dictionary file for np-wide properties",
        )
        parser.add_argument(
            "--freeze-molecule-encoder",
            action="store_true",
            default=False,
            help="freeze params in the molecule encoder module",
        )
        parser.add_argument(
            "--epoch-to-freeze-molecule-encoder",
            type=int,
            default=None,
            help="epoch to start freezing params in the molecule encoder module",
        )
        # Custom arg for multitask regularization (cagrad)
        parser.add_argument(
            "--multitask-reg",
            action="store_true",
            default=False,
            help="regularize multitask learning with cagrad (conflict-averse gradient descent)",
        )
        parser.add_argument(
            "--cagrad-c",
            type=float_range(0.0, 1.0),
            default=0.5,
            help="hyperparameter for cagrad that constraints the maximum distance between the g_0 (average task gradient) and the final gradient d, must be between [0,1)",
        )
        parser.add_argument(
            "--contrast-margin-coeff",
            type=float,
            default=0.0,
            help="hyperparameter for multiplier of margin values (margin of predict difference in sigmoid operation) in contrastive learning objective, so that samples in a pair whose values are farther apart should have predicted scores to be farther apart due to a larger value where the sigmoid inflection point is",
        )
        # Noise augmentation for percent in NP task
        parser.add_argument(
            "--percent-noise",
            type=float_range(0.0, 1.0), # 1 is max noise, same magnitude as percent value
            default=0.01,
            help="Proportion of `percent` value to add as noise; 1 (100%) is max noise which has same magnitude as the `percent` value",
        )
        parser.add_argument(
            "--percent-noise-type",
            type=str,
            default="normal_proportionate",
            choices=["normal_proportionate", "trunc_normal_proportionate", "uniform_proportionate"],
        )
        parser.add_argument(
            "--noise-augment-percent",
            action="store_true",
            default=False,
            help="Add noise augmentation to `percent` value",
        )

    def __init__(self, args, dictionary, component_type_dictionary=None):
        super().__init__()
        # make np_base_architecture
        base_architecture(args)

        # model that will extract molecular features
        """
        self.mol_model's forward input:
            self,
            src_tokens,
            src_distance,
            src_coord,
            src_edge_type, # different atom-atom pairs have different src_edge_type
            encoder_masked_tokens=None,
            features_only=False,
            classification_head_name=None,
            **kwargs
        """
        self.mol_model = UniMolModel(args, dictionary)

        # Freeze params in mol_model:
        if args.freeze_molecule_encoder:
            for child in self.mol_model.children():
                for param in child.parameters():
                    param.requires_grad = False

            print("Complete freezing params, args.freeze_molecule_encoder: ", args.freeze_molecule_encoder)

        self.args = args
        self.padding_idx = dictionary.pad()

        self._num_updates = None

        # to tokenize component_type
        self.component_type_dictionary = component_type_dictionary
        if self.component_type_dictionary is not None: 
            if type(self.component_type_dictionary) == dict:
                self.component_type_embed_tokens = nn.ModuleDict()
                for component_type in self.component_type_dictionary:
                    cur_component_type_dictionary = self.component_type_dictionary[component_type]['dictionary']
                    if 'embed_dim' in self.component_type_dictionary[component_type]:
                        component_type_embed_dim = self.component_type_dictionary[component_type]['embed_dim']
                    else:
                        component_type_embed_dim = args.component_types_embed_dim
                    component_type_embed = nn.Embedding(
                        len(cur_component_type_dictionary), component_type_embed_dim, padding_idx=cur_component_type_dictionary.pad()
                    )
                    
                    input_key_value = component_type + "s" # pluralize input names
                    self.component_type_embed_tokens[input_key_value] = component_type_embed
            else:
                self.component_type_embed_tokens = nn.Embedding(
                    len(self.component_type_dictionary), args.component_types_embed_dim, padding_idx=self.component_type_dictionary.pad()
                )

        self.padding_idx = dictionary.pad()

        if self.args.include_excipients:
            n_component_type = 2 # 1 for lipid percents and 1 for excipient concentration values, they will have different mul and bias terms in the gaussian layer, to account for different scales 
        else:
            n_component_type = 1 # or 4 for IP, HL, CHO, PEG 
        self.gbf = GaussianLayer(args.percent_embed_dim, n_component_type) # different edge types have different mul and bias terms
        

        # option to only use one cls_embedding for all tasks 
        # Add CLS dict here, to store unique [CLS] for each task (e.g. cell/organ targets)
        self.cls_embeddings = nn.ParameterDict()

        if type(self.component_type_dictionary) == dict:
            sum_component_types_embed_dim = 0
            for component_type in self.component_type_dictionary:
                if 'embed_dim' in self.component_type_dictionary[component_type]:
                    component_type_embed_dim = self.component_type_dictionary[component_type]['embed_dim']
                else:
                    component_type_embed_dim = args.component_types_embed_dim
                sum_component_types_embed_dim += component_type_embed_dim

            self.component_rep_dim = args.encoder_embed_dim + args.percent_embed_dim + sum_component_types_embed_dim
        else:
            self.component_rep_dim = args.encoder_embed_dim + args.percent_embed_dim + args.component_types_embed_dim

        # Create hidden state for [CLS] of LNP model
        # self.lnp_CLS_embed =  nn.Parameter(torch.zeros(self.component_rep_dim).normal_(mean=0.0, std=0.02))
        
        # use custom CLS token for > 1 tasks, in multitask training, and if self.args.cls_emb_config != 'single'
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

            print("full_labels_dict: ", full_labels_dict)
            self.task_schema = full_labels_dict
            label_schema_present = True

        elif args.tasks_schema_path is not None and '.json' in args.tasks_schema_path:
            with open(os.path.join(args.data, args.tasks_schema_path), 'r') as openfile:
                # Reading from json file
                task_schema = json.load(openfile)
                self.task_schema = task_schema
            label_schema_present = True

        if label_schema_present:
            if self.args.cls_emb_config == 'multi':
                for task_name in self.task_schema:
                    # print("NPUniMolModel __init__ task_name: ", task_name)
                    self.register_cls_embed(task_name, self.component_rep_dim)
            else: # self.args.cls_emb_config == 'single'
                self.register_cls_embed('default', self.component_rep_dim)
        else:
            self.task_schema = None
            self.register_cls_embed('default', self.component_rep_dim)

        # TODO NOW: Handle multi-task learning with multiple datasets
        # create modules for prop_dataset
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
            print("unimol.py self.np_prop_schema with full_dataset_task_schema_path: ", self.np_prop_schema)
        elif args.np_prop_schema_path is not None and '.json' in args.np_prop_schema_path:
            with open(os.path.join(args.data, args.np_prop_schema_path), 'r') as openfile:
                # Reading from json file
                np_prop_schema = json.load(openfile)
                self.np_prop_schema = np_prop_schema
        else:
            self.np_prop_schema = None

        if self.np_prop_schema != None:
            self.np_prop_embs = nn.ModuleDict()
            for prop in self.np_prop_schema:
                if self.np_prop_schema[prop]['type'] == 'continuous':
                    self.np_prop_embs[prop] = GaussianLayer(self.component_rep_dim, 1)

                elif self.np_prop_schema[prop]['type'] == 'categorical':
                    print("before embedding init, prop: ", prop, " self.np_prop_schema: ", self.np_prop_schema)
                    self.np_prop_embs[prop] = nn.Embedding(
                        len(self.np_prop_schema[prop]['dictionary']), self.component_rep_dim
                    )
        else:
            self.np_prop_embs = None            
        
        self.lnp_rep_proj = NonLinearHead(
            self.component_rep_dim, args.lnp_encoder_embed_dim, args.activation_fn, hidden=(self.component_rep_dim+args.lnp_encoder_embed_dim)//2
        )

        # Customize args for LNP encoder
        self.lnp_encoder = LNPTransformerEncoder(
            encoder_layers=args.lnp_encoder_layers, 
            embed_dim=args.lnp_encoder_embed_dim,
            ffn_embed_dim=args.lnp_encoder_ffn_embed_dim,
            attention_heads=args.lnp_encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            # no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )

        self.classification_heads = nn.ModuleDict()

        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        model = cls(args, task.dictionary, task.component_type_dictionary)
        return model

    def forward(
        self,
        input,
        # encoder_masked_tokens=None,
        # features_only=False,
        np_classification_head_name=None,
        # classification_head_name=None,
        **kwargs
    ):  
        mol_model_input = input['mol_features']

        encoder_rep, encoder_pair_rep = self.mol_model(
            **mol_model_input,
            features_only=True,
            output_rep_only=True,
        )

        # extract batch's NP's data
        np_model_input = input['net_input']

        # convert input data to reps

        # Convert mol_ids to rep
        mol_batch_ids = input['mol_batch_ids']
        mol_rep = encoder_rep[:, 0, :]
        
        mol_batch_ids_shape = mol_batch_ids.shape
        flattened_mol_batch_ids = mol_batch_ids.flatten()
        
        # add rep for pad_idx to mol_rep, so that we can index mol_rep with flattened_mol_batch_ids to also select pad_idx's rep
        pad_rep = torch.zeros_like(mol_rep[0]).unsqueeze(0)
        mol_rep = torch.cat([mol_rep, pad_rep], dim=0)
        flattened_mol_batch_ids[flattened_mol_batch_ids==-1] = mol_rep.shape[0] - 1 # replace -1 index values with largest index value to select pad_idx's rep, to avoid indexing error from index_select
        flattened_lnp_mol_rep = torch.index_select(mol_rep, 0, flattened_mol_batch_ids)
        lnp_mol_rep = torch.unflatten(flattened_lnp_mol_rep, 0, mol_batch_ids_shape)

        # Convert percents to rep 
        percents = np_model_input['percents']
        def get_percent_features(percent, c_type):

            gbf_feature = self.gbf(percent, c_type)
            return gbf_feature
        
        # new implementation that combines the composition values of lipid components and excipients together in np_model_input['percents']
        if self.args.include_excipients:
            if "composition_enc_types" not in np_model_input:
                raise RuntimeError("include_excipients is True but composition_enc_types is not provided in np_model_input")
            c_type = np_model_input['composition_enc_types'].type(torch.int64)
        else:
            c_type = torch.zeros_like(percents).type(torch.int64)
        # SSLNP_todo_end
        percents_rep = get_percent_features(percents, c_type) # idea: use percents_rep as graph_attn_bias

        component_types = np_model_input['component_types']
        if self.component_type_dictionary is not None:
            # to process if there is more than one component_type
            if type(self.component_type_dictionary) == dict:
                component_types_rep_dict = {}
                component_types_rep_list = []
                for input_key_value in self.component_type_embed_tokens: # self.component_type_embed_tokens (nn.ModuleDict) is ordered
                    component_types_input = np_model_input[input_key_value]
                    component_types_rep = self.component_type_embed_tokens[input_key_value](component_types_input)
                    component_types_rep_dict[input_key_value] = component_types_rep_dict
                    component_types_rep_list.append(component_types_rep)

                rep_list_to_cat = [lnp_mol_rep, percents_rep] + component_types_rep_list
                lnp_component_rep = torch.cat(rep_list_to_cat, dim=-1) # last dim: concat lnp_mol_rep, percents_rep and component_types_rep's, self.component_rep_dim
                
            else:
                component_types_rep = self.component_type_embed_tokens(component_types)
                lnp_component_rep = torch.cat([lnp_mol_rep, percents_rep, component_types_rep], dim=-1) # last dim: self.component_rep_dim 
        else:
            lnp_component_rep = torch.cat([lnp_mol_rep, percents_rep], dim=-1) # last dim: self.component_rep_dim 

        bsz = lnp_component_rep.shape[0]
        lnp_component_mask = torch.ones([bsz,lnp_component_rep.shape[1]]).bool().to(lnp_component_rep)

        
        # Create CLS token(s) that will be added to the start of mol_id sequence, to account for these additional tokens during inference
        if type(self.component_type_dictionary) == dict:
            cls_token = torch.full(component_types.size()[:-1], fill_value=self.component_type_dictionary['component_type']['dictionary'].bos())
        else:
            cls_token = torch.full(component_types.size()[:-1], fill_value=self.component_type_dictionary.bos())

        cls_token = cls_token.to(component_types, non_blocking=True)
        cls_token = cls_token.unsqueeze(-1)

        # Use custom CLS token for > 1 tasks, in multitask training
        # Add [CLS] hidden state to the start of the input reps, for each prediction target (e.g. organ/cell types)
        # Make task CLS embeds and add to lnplnp_component_rep

        # add prop CLS embeds to lnp_component_rep
        if self.np_prop_schema is not None:
            prop_embed_list = []
            prop_token_list = []
            prop_mask_list = []

            # Create mask for np_prop when input samples are not using all np_prop
            for prop in self.np_prop_schema:

                prop_value = np_model_input[prop]
                prop_mask_name = prop + "_mask"
                prop_mask = torch.unsqueeze(np_model_input[prop_mask_name], dim=-1)

                # handle categorical prop here
                if self.np_prop_schema[prop]['type'] == 'categorical':
                    prop_rep = self.np_prop_embs[prop](prop_value)
                    if len(prop_rep.shape) < 3:
                        for i in range(3 - len(prop_rep.shape)):
                            prop_rep = prop_rep.unsqueeze(dim=1)

                elif self.np_prop_schema[prop]['type'] == 'continuous':
                    def get_continuous_prop_features(value, c_type):
                        gbf_feature = self.np_prop_embs[prop](value, c_type)
                        return gbf_feature
                    c_type = torch.zeros_like(prop_value).type(torch.int64)
                    
                    prop_rep = get_continuous_prop_features(prop_value, c_type).unsqueeze(dim=1) # to get additional dim for seq len

                prop_embed_list.append(prop_rep) 
                prop_token_list.append(cls_token)
                prop_mask_list.append(prop_mask)

            lnp_component_mask_list = prop_mask_list + [lnp_component_mask]
            lnp_component_mask = torch.cat(lnp_component_mask_list, dim=1) 

            # add prop_embs' hidden state to the front of component rep list
            lnp_component_rep_list = prop_embed_list + [lnp_component_rep]
            lnp_component_rep = torch.cat(lnp_component_rep_list, dim=1) 
            
            # prepend cls_token to token sequence
            new_component_types_seq = prop_token_list + [component_types]
            component_types = torch.cat(new_component_types_seq, dim=1) 
        
        if self.task_schema == None or self.args.cls_emb_config == 'single': # use this case if only one cls_embedding is used for (all) tasks
            lnp_CLS_embed = self.cls_embeddings['default'].view(1, 1, -1).expand(bsz, -1, -1)
            lnp_component_rep = torch.cat([lnp_CLS_embed, lnp_component_rep], dim=1) # add [CLS] hidden state to the front of component rep list
            lnp_component_mask = torch.cat([torch.ones([bsz,lnp_CLS_embed.shape[1]]).bool().to(lnp_component_rep), lnp_component_mask], dim=1) # add [CLS] hidden state to the front of component rep list

            # prepend cls_token to token sequence
            component_types = torch.cat([cls_token, component_types], dim=1) 
        else:
            CLS_embed_list = []
            task_names = []
            cls_token_list = []
            for task_name in self.task_schema:
                task_CLS_embed = self.cls_embeddings[task_name].view(1, 1, -1).expand(bsz, -1, -1)
                CLS_embed_list.append(task_CLS_embed)
                task_names.append(task_name)
                cls_token_list.append(cls_token)

            lnp_component_mask_list = [ torch.ones([ bsz, len(CLS_embed_list) ]).bool().to(lnp_component_rep) ] + [lnp_component_mask]
            lnp_component_mask = torch.cat(lnp_component_mask_list, dim=1) 

            # add [CLS]s' hidden state to the front of component rep list
            lnp_component_rep_list = CLS_embed_list + [lnp_component_rep]
            lnp_component_rep = torch.cat(lnp_component_rep_list, dim=1) 

            # prepend cls_token to token sequence
            new_component_types_seq = cls_token_list + [component_types]
            component_types = torch.cat(new_component_types_seq, dim=1) 

        # Use input's component_type token to find pad in input sequences
        # padded input will be zeroed during inference
        if type(self.component_type_dictionary) == dict:
            padding_mask = component_types.eq(self.component_type_dictionary['component_type']['dictionary'].pad())
        else:
            padding_mask = component_types.eq(self.component_type_dictionary.pad())
        if not padding_mask.any():
            padding_mask = None

        # cross-component attention mechanism
        length = lnp_component_rep.shape[1]
        # placeholder attn mask for encoder input
        attn_mask_zeros = torch.zeros([bsz * self.lnp_encoder.attention_heads, length, length]).to(lnp_component_rep)
        
        # use attn_mask/attn_mask_zeros to mask out non-relevant task-specific cls tokens (e.g. N/P cls tokens for tasks that don't have N/P component)
        # USE lnp_component_mask (shape: [bsz, length]), dtype: bool

        x_lnp = self.lnp_rep_proj(lnp_component_rep)

        (
            lnp_encoder_rep,
            attn_weight,
            # delta_encoder_pair_rep,
            x_norm,
            # delta_encoder_pair_rep_norm,
        ) = self.lnp_encoder(x_lnp, padding_mask=padding_mask, attn_mask=attn_mask_zeros, lnp_component_mask=lnp_component_mask)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        # get logits for >1 tasks from classification head, logits would be a dict of logits if schema == dict
        if self.task_schema == None:
            if np_classification_head_name is not None and np_classification_head_name in self.classification_heads:
                logits = self.classification_heads[np_classification_head_name](lnp_encoder_rep)
                cls_representations = lnp_encoder_rep[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            logits = {}
            cls_representations = {}
            for task_ind, task_name in enumerate(self.task_schema):
                if self.args.cls_emb_config == 'single' and self.args.cls_head_config == 'multi':
                    # use the first cls embedding for all classification_heads' inference: cls_ind=0
                    task_cls_representation = lnp_encoder_rep[:, 0, :]  # take <s> token (equiv. to [CLS])
                    task_logits = self.classification_heads[task_name](lnp_encoder_rep, cls_ind=0)

                elif self.args.cls_emb_config == 'multi' and self.args.cls_head_config == 'single':
                    # use the first cls embedding for all classification_heads' inference: cls_ind=0
                    task_cls_representation = lnp_encoder_rep[:, task_ind, :]  # take <s> token (equiv. to [CLS])
                    task_logits = self.classification_heads[np_classification_head_name](lnp_encoder_rep, cls_ind=task_ind)
                    
                # when multiple cls heads and cls embs are used
                else:
                    if task_names[task_ind] != task_name:                    
                        logger.warning(
                            'task_name orders for cls and classification head are misaligned, task_name for cls: {} , task_name for head: {}'.format(
                                task_names[task_ind], task_name
                            )
                        )
                    task_cls_representation = lnp_encoder_rep[:, task_ind, :]  # take <s> token (equiv. to [CLS])
                    task_logits = self.classification_heads[task_name](lnp_encoder_rep, cls_ind=task_ind)

                cls_representations[task_name] = task_cls_representation
                logits[task_name] = task_logits

        # return final representation of [CLS] tokens here
        return (
            logits, cls_representations
        )         

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.lnp_encoder_embed_dim,
            inner_dim=inner_dim or self.args.lnp_encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def register_cls_embed(
        self, name, dim=None, mean=0.0, std=0.02, **kwargs
    ):
        """Register a CLS embedding, one for each task (e.g. cell/organ targets)."""
        if name in self.cls_embeddings:
            # prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_dim = self.cls_embeddings[name].dim
            if dim != prev_dim:
                logger.warning(
                    're-registering head "{}" with dim {} (prev: {})'.format(
                        name, dim, prev_dim
                    )
                )
        self.cls_embeddings[name] = nn.Parameter(torch.zeros(dim).normal_(mean=mean, std=std))

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


@register_model("unimol")
class UniMolModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--loss-sample-dropout", type=float, metavar="D", help="loss sample dropout probability"
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        # LNP model-related args
        parser.add_argument('--load_full_np_model', default=False, action='store_true',
                        help='load weights of full LNP model or only those of its mol_model')


    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type) # different edge types have different mul and bias terms

        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.encoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.encoder_attention_heads, args.activation_fn
            )
        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task=None, dictionary=None):
        """Build a new model instance."""
        # print("models/unimol.py's build_model")
        if task is not None:
            return cls(args, task.dictionary)
        else:
            return cls(args, dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        output_rep_only=False,
        **kwargs
    ):

        if classification_head_name is not None:
            features_only = True

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = NonLinearHead
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.lm_head(encoder_rep, encoder_masked_tokens)
            if self.args.masked_coord_loss > 0:
                coords_emb = src_coord
                if padding_mask is not None:
                    atom_num = (torch.sum(1 - padding_mask.type_as(x), dim=1) - 1).view(
                        -1, 1, 1, 1
                    )
                else:
                    atom_num = src_coord.shape[1] - 1
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                coord_update = torch.sum(coord_update, dim=2)
                encoder_coord = coords_emb + coord_update
            if self.args.masked_dist_loss > 0:
                encoder_distance = self.dist_head(encoder_pair_rep)

        if classification_head_name is not None and classification_head_name in self.classification_heads:
            logits = self.classification_heads[classification_head_name](encoder_rep)
        if self.args.mode == 'infer' or output_rep_only == True:
            return encoder_rep, encoder_pair_rep
        else:
            return (
                logits,
                encoder_distance,
                encoder_coord,
                x_norm,
                delta_encoder_pair_rep_norm,
            )         

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, cls_ind=0, **kwargs):
        if cls_ind == None:
            x = features[:]
        else:
            x = features[:, cls_ind, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        if len(x.shape) == 4:
            x = x.expand(-1, -1, -1, self.K)
        else:
            expand_list = [-1 for i in range(len(x.shape)-1)]
            expand_list.append(self.K)
            x = x.expand(expand_list)

        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


@register_model_architecture("np_unimol", "np_unimol")
def np_unimol_base_architecture(args):
    base_architecture(args)
    # add LNP cross-component attn model args
    args.lnp_encoder_layers = getattr(args, "lnp_encoder_layers", args.encoder_layers)
    args.lnp_encoder_embed_dim = getattr(args, "lnp_encoder_embed_dim", args.encoder_embed_dim)
    args.lnp_encoder_ffn_embed_dim = getattr(args, "lnp_encoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.lnp_encoder_attention_heads = getattr(args, "lnp_encoder_attention_heads", args.encoder_attention_heads)
    args.percent_embed_dim = getattr(args, "percent_embed_dim", 128)
    args.component_types_embed_dim = getattr(args, "component_types_embed_dim", 128)
    args.lnp_encoder_embed_dim = getattr(args, "lnp_encoder_embed_dim", args.encoder_embed_dim)
    args.cls_emb_config = getattr(args, "cls_emb_config", 'multi')
    args.cls_head_config = getattr(args, "cls_head_config", 'multi')
    # multitask regularization args
    args.multitask_reg = getattr(args, "multitask_reg", False)
    args.cagrad_c = getattr(args, "cagrad_c", 0.5)
    args.contrast_margin_coeff = getattr(args, "contrast_margin_coeff", 0.0)
    


@register_model_architecture("unimol", "unimol") # def register_model_architecture(model_name, arch_name), where model_name is the BaseUnicoreModel to use
def base_architecture(args):
    # print("base_architecture passed")
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.loss_sample_dropout = getattr(args, "loss_sample_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    # LNP model-related args
    args.load_full_np_model = getattr(args, "load_full_np_model", False)
    


@register_model_architecture("unimol", "unimol_base")
def unimol_base_architecture(args):
    print("unimol_base_architecture passed")
    base_architecture(args)
