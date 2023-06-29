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
Simulated world representations in DOROTHIE
"""
from dataclasses import dataclass
from enum import IntEnum, unique

from ontology.geometry import Transform, Rotation, WayPoint


@unique
class SemanticClass(IntEnum):
    """
    All the semantic classes in Carla
    """
    Unlabeled = 0
    Building = 1
    Fence = 2
    Other = 3
    Pedestrian = 4
    Pole = 5
    RoadLine = 6
    Road = 7
    SideWalk = 8
    Vegetation = 9
    Vehicles = 10
    Wall = 11
    TrafficSign = 12
    Sky = 13
    Ground = 14
    Bridge = 15
    RailTrack = 16
    GuardRail = 17
    TrafficLight = 18
    Static = 19
    Dynamic = 20
    Water = 21
    Terrain = 22
    Queried = 23


@unique
class TrafficLightState(IntEnum):
    """
    All the traffic light states
    """
    Unknown = 0
    Reg = 1
    Yellow = 2
    Green = 3
    Off = 4


@unique
class VehicleLightState(IntEnum):
    """
    All the traffic light states
    """
    Unknown = 0
    On = 1
    Off = 2


@dataclass
class Color:
    """
    Class that defines Carla color instance
    """
    r: int = 0
    g: int = 0
    b: int = 0
    a: int = 255


@dataclass
class BoundingBox:
    """
    Class that defines the bounding box of an object
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Control:
    """
    Class that defines animation parameters that controls the physics of vehicles
    """
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    handbrake: float = 0.0
    gear: int = 0
    reverse: bool = False
    manual_gear_shift: bool = False


@dataclass
class Speed:
    """
    Class that defines the speed of an agent
    """
    linear: Transform = Transform()
    angular: Rotation = Rotation()


@dataclass
class StaticObject:
    """
    Class that defines a vehicle
    """
    wp: WayPoint = WayPoint()
    extent: BoundingBox = BoundingBox()
    model: str = ''


@dataclass
class Pedestrian:
    """
    Class that defines a vehicle
    """
    speed: Speed = Speed()
    wp: WayPoint = WayPoint()
    extent: BoundingBox = BoundingBox()
    blueprint: str = ''


@dataclass
class Vehicle:
    """
    Class that defines a vehicle
    """
    control: Control = Control()
    speed: Speed = Speed()
    wp: WayPoint = WayPoint()
    extent: BoundingBox = BoundingBox()
    light: VehicleLightState = VehicleLightState.Off
    blueprint: str = ''


SEMANTIC_COLOR = {
    SemanticClass.Unlabeled: (0, 0, 0),
    SemanticClass.Building: (70, 70, 70),
    SemanticClass.Fence: (100, 40, 40),
    SemanticClass.Other: (55, 90, 80),
    SemanticClass.Pedestrian: (220, 20, 60),
    SemanticClass.Pole: (153, 153, 153),
    SemanticClass.RoadLine: (157, 234, 50),
    SemanticClass.Road: (128, 64, 128),
    SemanticClass.SideWalk: (244, 35, 232),
    SemanticClass.Vegetation: (107, 142, 35),
    SemanticClass.Vehicles: (0, 0, 142),
    SemanticClass.Wall: (102, 102, 156),
    SemanticClass.TrafficSign: (220, 220, 0),
    SemanticClass.Sky: (70, 130, 180),
    SemanticClass.Ground: (81, 0, 81),
    SemanticClass.Bridge: (150, 100, 100),
    SemanticClass.RailTrack: (230, 150, 140),
    SemanticClass.GuardRail: (180, 165, 180),
    SemanticClass.TrafficLight: (250, 170, 30),
    SemanticClass.Static: (110, 190, 160),
    SemanticClass.Dynamic: (170, 120, 50),
    SemanticClass.Water: (45, 60, 150),
    SemanticClass.Terrain: (145, 170, 100)
}
