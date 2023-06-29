#!/usr/bin/python3

# ==============================================================================
# -- COPYRIGHT -----------------------------------------------------------------
# ==============================================================================

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# Copyright (c) 2022 SLED Lab, EECS, University of Michigan.
# authors: Martin Ziqiao Ma (marstin@umich.edu),
#          Ben VanDerPloeg (bensvdp@umich.edu),
#          Owen Yidong Huang (owenhji@umich.edu),
#          Elina Eui-In Kim (euiink@umich.edu),
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random waypoints.
This agent avoids other vehicles and responds to traffic lights.
This agent does NOT respond to stop signs.
"""

# ==============================================================================
# -- IMPORTS -------------------------------------------------------------------
# ==============================================================================

import time
import carla
from agents.navigation.agent import Agent, AgentState
from co_wizard_local_planner import LocalPlanner


# ==============================================================================
# -- ROAMINGAGENT --------------------------------------------------------------
# ==============================================================================

class Co_WizardAgent(Agent):
    """
    Co_WizardAgent implements a basic agent that navigates scenes making random
    choices when facing an intersection.

    This agent avoids other vehicles and responds to traffic lights.
    This agent does NOT respond to stop signs.
    """

    def __init__(self, vehicle, map_input_queue=None, lpqueue=None):
        """
        Constructor for RoamingAgent class
        :param vehicle: actor to apply to local planner logic onto
        :param lpqueue: queue to relay keyboard commands to the local planner
        """
        super(Co_WizardAgent, self).__init__(vehicle)
        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self._local_planner = LocalPlanner(self._vehicle, map_input_queue=map_input_queue)

        self.vehicle = vehicle
        self.lpqueue = lpqueue
        self.stopsign_stopped_time = None
        self.stopsign_proceed = False

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        doLaneChange = 'NONE'

        while not self.lpqueue.empty():
            cmd, val = self.lpqueue.get()
            if cmd == 'brake':
                self._local_planner.set_brake(val)
            elif cmd == 'parked':
                self._local_planner.set_parked(val)
            elif cmd == 'clear_wps':
                print('waypoints cleared')
                self._local_planner.clear_wps()
            elif cmd == 'speedAdj':
                self._local_planner.speed_adj(val)
                print('New target speed: %d km/h' % self._local_planner._target_speed)
            elif cmd == 'pathOption':
                self._local_planner.select_path_option(val)
            elif cmd == 'laneChangeReq':
                doLaneChange = val
            elif cmd == 'planLaneChange':
                self._local_planner.queue_lane_change(val)
            elif cmd == 'uturn':
                self._local_planner.do_uturn()
            else:
                print('NOTE: unrecognized LP command %s' % cmd)

        # is there an obstacle in front of us?
        hazard = None

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            # if debug:
            print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard = ("blocked", vehicle.id)

        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            #if debug:
            print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard = ("red_light", traffic_light.id)

        if hazard is not None:
            control = self.emergency_stop()
            print('Emergency stopping')
        else:
            control = self._local_planner.run_step(doLaneChange=doLaneChange)
            self._state = AgentState.NAVIGATING

        return control, hazard
