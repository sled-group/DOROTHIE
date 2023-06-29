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


from typing import List
from argparse import Namespace

import torch
from torch import nn

from ontology.agent import PhysicalAction, PA_ARGS
from ontology.geometry import Direction
from ontology.carla import VehicleLightState


class ActionEncoder(nn.Module):

    def __init__(self, args: Namespace):
        """
        Encoder for Actions
        """
        super(ActionEncoder, self).__init__()

        map_name = args.config['environment']['map']
        self.road_id = args.metaconfig['map'][map_name]['road_id']

        self.encoder = nn.Embedding(len(PhysicalAction.__members__), args.train['d_emb'] - 1)

    def forward(self, actions: List[int], arguments: List[int]) -> torch.tensor:
        """
        pass action history through look up table and encode
        """
        act_emd = self.encoder(torch.tensor(actions))
        arg_emd = torch.tensor(arguments).unsqueeze(dim=1)
        return torch.cat((act_emd, arg_emd), dim=1)

    def sanity_check(self, action: PhysicalAction, argument: int):
        """
        Sanity check for action and argument
        """
        if PA_ARGS[action] is None:
            assert argument == 0
        elif PA_ARGS[action] in [Direction, VehicleLightState]:
            assert argument in [1, 2]
        elif action == PhysicalAction.JTurn:
            assert argument in self.road_id
