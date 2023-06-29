#!/usr/bin/python3

# ==============================================================================
# -- COPYRIGHT -----------------------------------------------------------------
# ==============================================================================

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Copyright (c) 2022 SLED Lab, EECS, University of Michigan.
# authors: Martin Ziqiao Ma (marstin@umich.edu),
#          Ben VanDerPloeg (bensvdp@umich.edu),
#          Owen Yidong Huang (owenhji@umich.edu),
#          Elina Eui-In Kim (euiink@umich.edu),
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List
from argparse import Namespace

import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertConfig, BertModel, BertPreTrainedModel


BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased"
]


class TextEncoder(BertPreTrainedModel):

    def __init__(self, config: BertConfig, num_feature: int):
        super().__init__(config)

        self.bert = BertModel(config).from_pretrained(config['tokenizer'])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.transform = nn.Linear(config.hidden_size, num_feature)
        self.init_weights()

        self.human_speaker_token = '[HUM]'
        self.bot_speaker_token = '[BOT]'

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
    ):

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        pass

    def mask(self):
        pass
        # TODO: Mask the [HUM] and [BOT] speaker roles
        # See: https://github.com/alexpashevich/E.T./blob/master/alfred/nn/enc_lang.py

