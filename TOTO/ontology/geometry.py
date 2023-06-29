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

"""
Navigation Geometry in DOROTHIE
"""
from enum import IntEnum, unique
from dataclasses import dataclass
from typing import List


@unique
class Direction(IntEnum):
    """
    Represents a direction in the world.
    """
    Unknown = 0
    Left = 1
    Right = 2
    Front = 3
    Back = 4


@dataclass
class Location:
    """
    Represents a spot in the world.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __init__(self, dictionary=None):
        if dictionary is None:
            dictionary = {}
        for k, v in dictionary.items():
            setattr(self, k, v)

    def to_list(self):
        return [
            self.x,
            self.y,
            self.z,
        ]

    def __len__(self):
        return 3


@dataclass
class Rotation:
    """
    Class that defines a 3D orientation in space.
    """
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

    def __init__(self, dictionary=None):
        if dictionary is None:
            dictionary = {}
        for k, v in dictionary.items():
            setattr(self, k, v)

    def to_list(self):
        return [
            self.pitch,
            self.yaw,
            self.roll,
        ]

    def __len__(self):
        return 3


@dataclass
class Transform:
    """
    Class that defines a transformation, a combination of location and rotation, without scaling.
    """
    location: Location = Location()
    rotation: Rotation = Rotation()

    def __init__(self, dictionary=None):
        if dictionary is not None:
            self.location = Location(dictionary['location'])
            self.rotation = Rotation(dictionary['rotation'])

    def to_list(self):
        return self.location.to_list() + self.rotation.to_list()

    def __len__(self):
        return len(self.location) + len(self.rotation)


@dataclass
class OpenDriveID:
    """
    Class that defines OpenDrive identifiers of road, lane, junction
    """
    lane_id: int = 0
    road_id: int = 0
    junction_id: int = 0

    def __init__(self, dictionary=None):
        if dictionary is None:
            dictionary = {}
        for k, v in dictionary.items():
            setattr(self, k, v)

    def to_list(self):
        return [
            self.lane_id,
            self.road_id,
            self.junction_id
        ]

    def __eq__(self, identifier):
        assert type(self) == type(identifier)
        return self.road_id == identifier.road_id and self.lane_id == identifier.lane_id

    def __len__(self):
        return 3


@dataclass
class WayPoint:
    """
    Class that defines a 3D directed point
    """
    transform: Transform = Transform()
    identifier: OpenDriveID = OpenDriveID()
    s: float = 0.0

    def to_list(self):
        return self.transform.to_list() + [self.s] + self.identifier.to_list()

    def __len__(self):
        return len(self.transform) + len(self.identifier) + 1


@dataclass
class Lane:
    """
    Class that defines a lane node in the topological graph
    """
    start: Transform = Transform()
    end: Transform = Transform()
    identifier: OpenDriveID = OpenDriveID()
    length: float = 0.0

    def __init__(self, dictionary=None):
        if dictionary is not None:
            self.start = Transform(dictionary['start'])
            self.end = Transform(dictionary['end'])
            self.identifier = OpenDriveID(dictionary['opendriveid'])
            self.length = dictionary['length']

    def to_list(self):
        return self.start.to_list() + self.end.to_list() + [self.length] + self.identifier.to_list()

    def __len__(self):
        return len(self.start) + len(self.end) + len(self.identifier) + 1


@dataclass
class Landmark:
    """
    Class that defines a landmark in a map
    """
    transform: Transform = Transform()
    name: str = ''
    asset: str = ''
    is_hidden: bool = False

    def __init__(self, dictionary=None):
        if dictionary is not None:
            self.transform = Transform(dictionary['transform'])
            self.identifier = OpenDriveID(dictionary['opendriveid'])
            self.asset = dictionary['asset']
            self.name = dictionary['name']

    def to_list(self):
        return self.transform.to_list() + self.identifier.to_list()

@dataclass
class StreetName:
    """
    Class that defines a street name in a map
    """
    name: str = ''
    lanes: List[int] = None
