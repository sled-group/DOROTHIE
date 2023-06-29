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
from typing import List

import torch
from torch import nn
from torch_sparse.tensor import SparseTensor

from nn.graph_encoder import MapEncoder
from nn.knowledge_encoder import OpenDriveEncoder, LandmarkEncoder, StreetNameEncoder

from ontology.geometry import Lane, Landmark, StreetName


class SemanticMemory(nn.Module):

    def __init__(self, args: Namespace, id_encoder: OpenDriveEncoder = None):
        """
        Encoder for long term semantic memory
        """
        super(SemanticMemory, self).__init__()

        self.id_encoder = OpenDriveEncoder(args, args.train['d_emb'] - len(Lane()) + 2) \
            if id_encoder is None else id_encoder
        self.map_encoder = MapEncoder(args, self.id_encoder, dropout=0.0, num_heads=8, activation='relu')
        self.landmark_encoder = LandmarkEncoder(args, self.id_encoder)
        self.street_encoder = StreetNameEncoder(args)

    def forward(self, lanes: List[Lane],
                edge_index: SparseTensor,
                landmarks: List[Landmark],
                streets: List[StreetName]) -> torch.tensor:
        """
        pass lanes through look up table and encode
        """
        map_emb = self.map_encoder(lanes, edge_index)
        lm_emb = self.landmark_encoder(landmarks)
        str_emb = self.street_encoder(streets)
        return torch.cat((map_emb, lm_emb, str_emb), dim=1)
