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
import torch.nn.functional as F
# from torch_geometric.nn import GATConv
# from torch_sparse.tensor import SparseTensor

from ontology.geometry import Lane
from nn.knowledge_encoder import OpenDriveEncoder
from nn.gat import GATLayerTorch


class LaneEncoder(nn.Module):

    def __init__(self,
                 num_feature: int,
                 id_encoder: OpenDriveEncoder = None):
        """
        Encoder for Lane
        """
        super(LaneEncoder, self).__init__()

        self.id_encoder = OpenDriveEncoder(num_feature - len(Lane()) + 2) \
            if id_encoder is None else id_encoder

    def forward(self, lanes: List[Lane], device='cpu') -> torch.tensor:
        """
        pass lanes through look up table and encode
        """
        lanes = torch.tensor([lane.to_list() for lane in lanes]).to(device)
        id_emb = self.id_encoder(lanes[:, -2].to(dtype=torch.int), lanes[:, -1].to(dtype=torch.int), device)
        return torch.cat((lanes[:, :-2], id_emb), dim=1)


class MapEncoder(nn.Module):

    def __init__(self,
                 num_feature: int,
                 id_encoder: OpenDriveEncoder = None,
                 dropout: float = 0.0,
                 num_heads: int = 8,
                 activation: str = 'relu'):
        """
        Encoder for Map Topology
        """
        super(MapEncoder, self).__init__()

        self.node_encoder = LaneEncoder(num_feature, id_encoder)
        self.topology_encoder = GATLayerTorch(num_feature, num_feature//num_heads,
                                              num_of_heads=num_heads, dropout_prob=dropout)
        self.activation = getattr(F, activation)

    def forward(self, lanes: List[Lane], edge_index: torch.Tensor, device='cpu') -> torch.Tensor:
        """
        pass map topology through look up table and graph attention layer and encode
        """
        x = self.node_encoder(lanes, device)
        return self.activation(self.topology_encoder(x, edge_index))


# class StreetNameEncoder(nn.Module):
#
#     def __init__(self, args: Namespace):
#         """
#         Encoder for Street names
#         """
#         super(StreetNameEncoder, self).__init__()
#
#     def forward(self, streets: List[StreetName]) -> torch.tensor:
#         """
#         pass street names through look up table and encode
#         """
#         # TODO: Need language models to encode the names of street names
#         pass
