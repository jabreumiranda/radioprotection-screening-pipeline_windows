# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

import numpy as np
import json

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        bos="[CLS]", # bos: beginning of sentence
        pad="[PAD]",
        eos="[SEP]",
        unk="[UNK]",
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.specials = set()
        self.specials.add(bos)
        self.specials.add(unk)
        self.specials.add(pad)
        self.specials.add(eos)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def vec_index(self, a):
        return np.vectorize(self.index)(a)

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.indices[self.unk_word]
    
    def special_index(self):
        return [self.index(x) for x in self.specials]

    def add_symbol(self, word, n=1, overwrite=False, is_special=False):
        """Adds a word to the dictionary"""
        if is_special:
            self.specials.add(word)
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.index(self.bos_word)

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.index(self.pad_word)

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.index(self.eos_word)

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.index(self.unk_word)

    @classmethod
    def load_from_json(cls, f):
        """Loads the dictionary from a json file with the format:

        ```
        {
            <component_type>: {
                "dictionary": [
                    "[PAD]", "[CLS]", "[SEP]", "[UNK]", <tok1>, <tok2>, <tok3>
                ],
                "embed_dim": <dim>,
            }
            ...
        }
        ```
        """
        # print("f: ", f)
        with open(f, 'r') as jsonfile:
            json_dict = json.load(jsonfile)

        dictionaries = {}
        # print("json_dict: ", json_dict)
        for component_type in json_dict:
            # print("component_type: ", component_type)
            dictionary = json_dict[component_type]['dictionary']

            d = cls()
            d.add_from_list(dictionary)

            dictionaries[component_type] = {}
            dictionaries[component_type]['dictionary'] = d
            
            if 'embed_dim' in json_dict[component_type]:
                dictionaries[component_type]['embed_dim'] = json_dict[component_type]['embed_dim']

        return dictionaries

    @classmethod
    def load_from_list(cls, l):
        """
        Loads the dictionary from a list of tokens
        """
        d = cls()
        d.add_from_list(l)

        return d

    @classmethod
    def load_component_types_from_master_schema_json(cls, f, key_name='np_component_types'):
        """Loads the dictionary from a master schema json file with the format:

        ```
        {
            ...
            np_component_types: {
                <component_type>: {
                    "dictionary": [
                        "[PAD]", "[CLS]", "[SEP]", "[UNK]", <tok1>, <tok2>, <tok3>
                    ],
                    "embed_dim": <dim>,
                }
                ...
            }    
        }
        ```
        """
        # print("f: ", f)
        with open(f, 'r') as jsonfile:
            json_dict = json.load(jsonfile)

        dictionaries = {}
        # print("json_dict: ", json_dict)
        dataset_component_types_dict = json_dict[key_name] # default: np_component_types
        for component_type in dataset_component_types_dict:
            # print("component_type: ", component_type)
            dictionary = dataset_component_types_dict[component_type]['dictionary']

            d = cls()
            d.add_from_list(dictionary)

            dictionaries[component_type] = {}
            dictionaries[component_type]['dictionary'] = d
            
            if 'embed_dim' in dataset_component_types_dict[component_type]:
                dictionaries[component_type]['embed_dim'] = dataset_component_types_dict[component_type]['embed_dim']

        return dictionaries
    
    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()

        for line_idx, line in enumerate(lines):
            try:
                splits = line.rstrip().rsplit(" ", 1)
                line = splits[0]
                field = splits[1] if len(splits) > 1 else str(len(lines) - line_idx)
                if field == "#overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                # print(line_idx, "word: ", word)
                if word in self and not overwrite:
                    logger.info(
                        "Duplicate word found when loading Dictionary: '{}', index is {}.".format(word, self.indices[word])
                    )
                else:
                    # print("add_from_file, add_symbol(word..: ", word)
                    self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

    def add_from_list(self, ls):
        """
        Loads a pre-existing dictionary from a list and adds its symbols
        to this instance.
        """

        for ele_idx, ele in enumerate(ls):
            try:
                # print(ele_idx, "ele: ", ele)
                if type(ele) != str:
                    ele = str(ele)
                splits = ele.rstrip().rsplit(" ", 1)
                ele = splits[0]
                field = splits[1] if len(splits) > 1 else str(len(ls) - ele_idx)
                # else:
                #     field = str(len(ls) - ele_idx)
                if field == "#overwrite":
                    overwrite = True
                    ele, field = ele.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = ele
                # print(ele_idx, "word: ", word)
                if word in self and not overwrite:
                    logger.info(
                        "Duplicate word found when loading Dictionary: '{}', index is {}.".format(word, self.indices[word])
                    )
                else:
                    # print("add_from_list, add_symbol(word..: ", word)
                    self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )