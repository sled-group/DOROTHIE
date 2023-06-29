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
Agent move space in DOROTHIE
"""
from enum import IntEnum, unique
from typing import List, Tuple

from dataclasses import dataclass

from ontology.geometry import Direction, Landmark
from ontology.carla import VehicleLightState


@unique
class PhysicalAction(IntEnum):
    """
    The physical action space of the agent
    """
    Unknown = 0
    LaneFollow = 1
    LaneSwitch = 2
    UTurn = 3
    JTurn = 4
    Stop = 5
    Start = 6
    SpeedChange = 7
    LightChange = 8
    Queried = 9


PA_ARGS = {
    PhysicalAction.Unknown: None,
    PhysicalAction.LaneFollow: None,
    PhysicalAction.LaneSwitch: Direction,
    PhysicalAction.UTurn: None,
    PhysicalAction.JTurn: int,
    PhysicalAction.Stop: None,
    PhysicalAction.Start: None,
    PhysicalAction.SpeedChange: int,
    PhysicalAction.LightChange: VehicleLightState
}


@unique
class Landmarks(IntEnum):
    """
    The landmarks
    """
    Unknown = 0
    BurgerKing = 1
    Coco = 2
    Ikea = 3
    KFC = 4
    Panera = 5
    Qdoba = 6
    SevenEleven = 7
    Shell = 8
    House = 9
    Person = 10
    Queried = 11

@unique
class StreetNames(IntEnum):
    """
    The goal status space of the agent
    """
    Unknown = 0
    Baits = 1
    Beal = 2
    Bishop = 3
    Bonisteel = 4
    Broadway = 5
    Division = 6
    Draper = 7
    Duffield = 8
    Fuller = 9
    Hayward = 10
    Hubbard = 11
    Murfin = 12
    Plymouth = 13
    Upland = 14
    Queried = 15
    Highway = 16
    
@unique
class GoalStatus(IntEnum):
    """
    The goal status space of the agent
    """
    Unknown = 0
    Ongoing = 1
    Complete = 2
    Abandoned = 3
    Pending = 4
    Queried = 5




@dataclass
class Goal:
    """
    Class that defines a navigational goal
    """
    landmark: Landmark = Landmark()
    context: str = ''
    delete: bool = False
    trigger: bool = False
    change: bool = False
    status: GoalStatus = GoalStatus.Unknown


@unique
class BeliefUpdate(IntEnum):
    """
    The belief update space of the agent
    """
    Unknown = 0
    PlanUpdate = 1
    GoalUpdate = 2
    StatusUpdate = 3
    LandmarkUpdate = 4
    Other = 5


BU_ARGS = {
    BeliefUpdate.Unknown: None,
    BeliefUpdate.PlanUpdate: List[PhysicalAction],
    BeliefUpdate.GoalUpdate: List[Goal],
    BeliefUpdate.StatusUpdate: Tuple[Goal, GoalStatus],
    BeliefUpdate.LandmarkUpdate: Landmark,
    BeliefUpdate.Other: None
}


@unique
class DialogueMove(IntEnum):
    """
    The dialogue move space of the agent
    """
    Irrelevant = 0
    Instruct = 1
    Explain = 2
    Align = 3
    Check = 4
    QueryYN = 5
    QueryW = 6
    Acknowledge = 7
    Confuse = 8
    Clarify = 9
    Confirm = 10
    ReplyY = 11
    ReplyN = 12
    ReplyU = 13
    ReplyW = 14


@unique
class DialogueSlot(IntEnum):
    """
    The dialogue slot space of the agent
    """
    Null = 0
    Action = 1
    Street = 2
    Landmark = 3
    Status = 4
    Object = 5


class DialogueMoveCodingScheme:
    """
    The dialogue move coding schedule of the agent
    """

    def __init__(self):
        # Is the utterance an initiation, response, or irrelevant?
        self.game_type = ['Irrelevant', 'Initiate', 'Respond']

        # Is the utterance a command, statement, or question?
        self.initiation_type = ['Instruct', 'Explain', 'Ask']

        # Is the utterance checking the success of communication, an inferrable fact, or querying for information?
        self.question_type = ['Align', 'Check', 'Query']

        # Is the query simple or complex
        self.query_type = ['Query-YN', 'Query-W']

        # Does the utterance indicate communication status of contribute to domain information?
        self.response_type = ['Communicate', 'Inform']

        # Does the utterance indicate successful communication or confusion
        self.communication_type = ['Acknowledge', 'Confuse']

        # Is the information amplified?
        self.information_type = ['Reply', 'Clarify']

        # Is the reply simple or complex
        self.reply_type = ['Reply-Y', 'Reply-N', 'Reply-W']

    def decision_tree(self, decisions: List[int]) -> Tuple[DialogueMove, DialogueSlot]:
        """
        The decision tree to decide the dialogue move and slot
        """
        try:

            assert len(decisions) == 9

            game_type, initiation_type, question_type, query_type, \
            response_type, communication_type, information_type, reply_type, slot = decisions

            assert 0 <= game_type <= len(self.game_type)
            assert 0 <= initiation_type <= len(self.initiation_type)
            assert 0 <= question_type <= len(self.question_type)
            assert 0 <= query_type <= len(self.query_type)
            assert 0 <= response_type <= len(self.response_type)
            assert 0 <= communication_type <= len(self.communication_type)
            assert 0 <= information_type <= len(self.information_type)
            assert 0 <= reply_type <= len(self.reply_type)
            assert 0 <= slot <= len(DialogueSlot.__members__)

            if game_type == 0:
                return DialogueMove.Irrelevant, DialogueSlot.Null

            elif game_type == 1:

                if initiation_type == 0:
                    assert DialogueSlot(slot) in DM_SLOT[DialogueMove.Instruct]
                    return DialogueMove.Instruct, DialogueSlot(slot)

                elif initiation_type == 1:
                    assert DialogueSlot(slot) in DM_SLOT[DialogueMove.Explain]
                    return DialogueMove.Explain, DialogueSlot(slot)

                elif question_type == 0:
                    return DialogueMove.Align, DialogueSlot.Null

                elif question_type == 1:
                    return DialogueMove.Check, DialogueSlot(slot)

                elif query_type == 0:
                    return DialogueMove.QueryYN, DialogueSlot.Null

                else:
                    return DialogueMove.QueryW, DialogueSlot(slot)

            elif response_type == 0:

                if communication_type == 0:
                    return DialogueMove.Acknowledge, DialogueSlot.Null

                if communication_type == 1:
                    return DialogueMove.Confuse, DialogueSlot.Null

            elif information_type == 0:

                if reply_type == 0:
                    return DialogueMove.ReplyY, DialogueSlot.Null

                elif reply_type == 1:
                    return DialogueMove.ReplyN, DialogueSlot.Null

                else:
                    return DialogueMove.ReplyW, DialogueSlot(slot)

            else:
                return DialogueMove.Clarify, DialogueSlot(slot)

        except AssertionError:
            return DialogueMove.Irrelevant, DialogueSlot.Null


DM_HIERARCHY = {
    DialogueMove.Irrelevant: ['Irrelevant'],
    DialogueMove.Instruct: ['Initiate', 'Instruct'],
    DialogueMove.Explain: ['Initiate', 'Explain'],
    DialogueMove.Align: ['Initiate', 'Ask', 'Align'],
    DialogueMove.Check: ['Initiate', 'Ask', 'Check'],
    DialogueMove.QueryYN: ['Initiate', 'Ask', 'Query', 'QueryYN'],
    DialogueMove.QueryW: ['Initiate', 'Ask', 'Query', 'QueryW'],
    DialogueMove.Acknowledge: ['Respond', 'Communicate', 'Acknowledge'],
    DialogueMove.Confuse: ['Respond', 'Communicate', 'Confuse'],
    DialogueMove.Clarify: ['Respond', 'Inform', 'Clarify'],
    DialogueMove.ReplyY: ['Respond', 'Inform', 'Reply', 'ReplyY'],
    DialogueMove.ReplyN: ['Respond', 'Inform', 'Reply', 'ReplyN'],
    DialogueMove.ReplyW: ['Respond', 'Inform', 'Reply', 'ReplyW']
}


DM_SLOT = {
    DialogueMove.Irrelevant: [DialogueSlot.Null],
    # DialogueMove.Instruct: [DialogueSlot.Action, DialogueSlot.Goal, DialogueSlot.Other],
    DialogueMove.Instruct: [DialogueSlot.Action],
    DialogueMove.Explain: [DialogueSlot.Status, DialogueSlot.Landmark,
                        #    DialogueSlot.Object, DialogueSlot.State, DialogueSlot.Other],
                           DialogueSlot.Object],
    DialogueMove.Align: [DialogueSlot.Null],
    # DialogueMove.Check: [DialogueSlot.Action, DialogueSlot.Goal,
    DialogueMove.Check: [DialogueSlot.Action, 
                         DialogueSlot.Status, DialogueSlot.Landmark,
                        #    DialogueSlot.Object, DialogueSlot.State, DialogueSlot.Other],
                           DialogueSlot.Object],
    DialogueMove.QueryYN: [DialogueSlot.Null],
    # DialogueMove.QueryW: [DialogueSlot.Action, DialogueSlot.Goal,
    DialogueMove.QueryW: [DialogueSlot.Action, 
                          DialogueSlot.Status, DialogueSlot.Landmark,
                        #    DialogueSlot.Object, DialogueSlot.State, DialogueSlot.Other],
                           DialogueSlot.Object],
    DialogueMove.Acknowledge: [DialogueSlot.Null],
    DialogueMove.Confuse: [DialogueSlot.Null],
    # DialogueMove.Clarify: [DialogueSlot.Action, DialogueSlot.Goal,
    DialogueMove.Clarify: [DialogueSlot.Action,
                           DialogueSlot.Status, DialogueSlot.Landmark,
                        #    DialogueSlot.Object, DialogueSlot.State, DialogueSlot.Other],
                           DialogueSlot.Object],
    DialogueMove.ReplyY: [DialogueSlot.Null],
    DialogueMove.ReplyN: [DialogueSlot.Null],
    # DialogueMove.ReplyW: [DialogueSlot.Action, DialogueSlot.Goal,
    DialogueMove.ReplyW: [DialogueSlot.Action, 
                          DialogueSlot.Status, DialogueSlot.Landmark,
                        #    DialogueSlot.Object, DialogueSlot.State, DialogueSlot.Other],
                           DialogueSlot.Object],
}
