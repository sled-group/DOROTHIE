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

from ontology.geometry import Lane, Landmark, StreetName


class RoadEncoder(nn.Module):

    def __init__(self,
                #  all_road_id: List[int],
                 num_roads: int,
                 num_feature: int):
        """
        Encoder for Road ID
        """
        super(RoadEncoder, self).__init__()

        # self.all_road_id = all_road_id
        self.num_roads = num_roads
        self.road_encoder = nn.Embedding(self.num_roads, num_feature)

    def forward(self, road_ids: List[int], device='cpu') -> torch.tensor:
        """
        pass identifiers through look up table and encode
        """
        return self.road_encoder(road_ids).to(device)

    # def retrieve(self, road_id: int):
    #     """
    #     retrieve the original road_id
    #     """
    #     return self.all_road_id[road_id]


class JunctionEncoder(nn.Module):

    def __init__(self,
                #  all_junction_id: List[int],
                 num_junctions: int,
                 num_feature: int):
        """
        Encoder for Junction ID
        """
        super(JunctionEncoder, self).__init__()

        # self.all_junction_id = all_junction_id
        self.num_junctions = num_junctions
        self.junction_encoder = nn.Embedding(self.num_junctions, num_feature)

    def forward(self, junction_ids: List[int], device='cpu') -> torch.tensor:
        """
        pass identifiers through look up table and encode
        """
        return self.junction_encoder(junction_ids).to(device)

    # def retrieve(self, junction_id: int):
    #     """
    #     retrieve the original junction_id
    #     """
    #     return self.all_junction_id[junction_id]


class OpenDriveEncoder(nn.Module):

    def __init__(self,
                 num_feature: int,
                #  all_road_id: List[int] = None,
                #  all_junction_id: List[int] = None,
                 num_roads: int,
                 num_junctions: int,
                 road_encoder: RoadEncoder = None,
                 junction_encoder: JunctionEncoder = None):
        """
        Encoder for OpenDrive ID
        """
        super(OpenDriveEncoder, self).__init__()

        # self.all_road_id = all_road_id
        # self.all_junction_id = all_junction_id
        self.num_roads = num_roads
        self.num_junctions = num_junctions

        self.road_encoder = nn.Embedding(self.num_roads, num_feature) \
            if road_encoder is None else road_encoder
        self.junction_encoder = nn.Embedding(self.num_junctions, num_feature) \
            if junction_encoder is None else junction_encoder

    def forward(self, road_ids: List[int], junction_ids: List[int], device='cpu') -> torch.tensor:
        """
        pass identifiers through look up table and encode
        """
        road_emd = self.road_encoder(road_ids,device)
        junction_emb = self.junction_encoder(junction_ids,device)

        return torch.cat([road_emd , junction_emb],dim=1)


class LandmarkEncoder(nn.Module):

    def __init__(self,
                 num_feature: int,
                 id_encoder: OpenDriveEncoder = None):
        """
        Encoder for Landmarks
        """
        super(LandmarkEncoder, self).__init__()

        self.id_encoder = OpenDriveEncoder(num_feature - len(Lane()) + 2) \
            if id_encoder is None else id_encoder

    def forward(self, landmarks: List[Landmark], device='cpu') -> torch.tensor:
        """
        pass landmark through look up table and encode
        """
        landmarks = torch.tensor([landmark.to_list() for landmark in landmarks]).to(device)
        id_emb = self.id_encoder(landmarks[:, -2].to(dtype=torch.int), landmarks[:, -1].to(dtype=torch.int), device)
        return torch.cat((landmarks[:, :-2], id_emb), dim=1)

