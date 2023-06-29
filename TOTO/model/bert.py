#!/usr/bin/python3

# ==============================================================================
# -- COPYRIGHT -----------------------------------------------------------------
# ==============================================================================

# Copyright (c) 2022 SLED Lab, EECS, University of Michigan.
# authors: Martin Ziqiao Ma (marstin@umich.edu),
#          Cristian-Paul Bara (cpbara@umich.edu),
#          Owen Yidong Huang (owenhji@umich.edu),
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


from argparse import Namespace
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from transformers import Wav2Vec2Processor, HubertModel
from transformers import DetrModel, DetrFeatureExtractor

class LinearLayer(nn.Module):

    def __init__(self, args: Namespace):
        super(LinearLayer, self).__init__()
        self.args = args
        self.movelayer=nn.Linear(768,15)
        self.typelayer=nn.Linear(768,5)
        self.slotlayer=nn.Linear(768,69)




    # def forward(self, dlg_emb, dlg_move_emb, act_emb, knowledge_emb, rgb, obj_detr, obj_def_detr, speech, belief_emb, sincos):
    def forward(self, x):

        return self.movelayer(x),self.typelayer(x),self.slotlayer(x)

        pass