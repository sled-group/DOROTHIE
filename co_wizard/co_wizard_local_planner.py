#!/usr/bin/python3

# ==============================================================================
# -- COPYRIGHT -----------------------------------------------------------------
# ==============================================================================

# Copyright (c) 2018-2020 CVC.
#
# Copyright (c) 2022 SLED Lab, EECS, University of Michigan.
# authors: Martin Ziqiao Ma (marstin@umich.edu),
#          Ben VanDerPloeg (bensvdp@umich.edu),
#          Owen Yidong Huang (owenhji@umich.edu),
#          Elina Eui-In Kim (euiink@umich.edu),
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

from enum import Enum
from collections import deque
import random

import carla
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import draw_waypoints
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from carla import Color
import math
import sys
from numpy import sign


MAX_TRAJ_LEN = 8
WP_COLOR = Color(0, 255, 0, 255)


def draw_waypoints_color(world, waypoints, z=0.5, color=WP_COLOR):
    """
    Draw a list of waypoints at a certain height given in z.
    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :param color: waypoint color
    """
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0, color=color)


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to another.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None, map_input_queue=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F
            target_speed -- desired cruise speed in Km/h
            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead
            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}
            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self._vehicle = vehicle
        self.world = self._vehicle.get_world()
        self._map = self.world.get_map()
        self.map_input_queue = map_input_queue

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None

        self.goal = None

        # low-level waypoint queue to next junction, with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque()

        self.pathSelection = None

        # list of waypoints for entire current path, for rendering purposes only.
        # may be inaccurate due to unknown lane change timing, so we don't use for navigation

        self.nextOptions = None

        self.manual_brake = False
        self.parked = False

        # self.uturn_new_lane_id = None
        self.opt_colors = [carla.Color(*(x + (255,)))
                           for x in [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                                     (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]]

        # initializing controller
        self._init_controller(opt_dict)
        global_planner_dao = GlobalRoutePlannerDAO(self._map, self._sampling_radius)
        self._global_planner = GlobalRoutePlanner(global_planner_dao)

        self.get_next_turn_options()

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
            print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 30.0  # 30 fps
        self._target_speed = 30.0  # Km/h
        self._sampling_radius = 5.0  # self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        self._max_brake = 0.3
        self._max_throt = 0.75
        self._max_steer = 0.8

        args_lateral_dict = {
            'K_P': 0.8,  # 1.95,
            'K_D': 0.0,  # 0.2,
            'K_I': 0.0,  # 0.07
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 0.1,
            'K_D': 0,
            'K_I': 0.005,
            'dt': self._dt}

        self._offset = 0

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                                        opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']
            if 'max_throttle' in opt_dict:
                self._max_throt = opt_dict['max_throttle']
            if 'max_brake' in opt_dict:
                self._max_brake = opt_dict['max_brake']
            if 'max_steering' in opt_dict:
                self._max_steer = opt_dict['max_steering']
            if 'offset' in opt_dict:
                self._offset = opt_dict['offset']

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)


    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def get_next_turn_options(self,need_report_new_road=True):
        road_list=[]
        while 1:
            if self._waypoints_queue:
                startwp = self._waypoints_queue[-1][0]
            else:
                startwp = self._map.get_waypoint(self._vehicle.get_location())

            for wp in startwp.next_until_lane_end(self._sampling_radius):
                self._waypoints_queue.append((wp, RoadOption.LANEFOLLOW))

            turnopts = self._waypoints_queue[-1][0].next(self._sampling_radius)
            # Only one option, just continue
            if len(turnopts) == 1:
                road_id=turnopts[0].road_id
                if road_id in road_list:
                    break
                road_list.append(road_id)

                self._waypoints_queue.append((turnopts[0], RoadOption.LANEFOLLOW))
            else:
                break
        if len(turnopts)==1:
            print('only one turn options')
            return 

        opts = _retrieve_options(turnopts, startwp)
        self.nextOptions = [(wp, opts[i]) for i, wp in enumerate(turnopts)]
        print('found %d turn options' % len(self.nextOptions))

        self.map_input_queue.put(('path_options',
                               [[[pt.transform.location.x, pt.transform.location.y, pt.road_id]
                                 for pt in st.next_until_lane_end(self._sampling_radius)[:20]]
                                for st, _ in self.nextOptions]))
        if need_report_new_road:
            self.map_input_queue.put(('new_road_id', self._waypoints_queue[0][0].road_id))

    def run_step(self, doLaneChange='NONE', debug=False):
        """current_waypoint
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return: control to be applied
        """

        # current vehicle waypoint
        vehicle_transform = self._vehicle.get_transform()
        self._current_waypoint = self._map.get_waypoint(vehicle_transform.location)
        if doLaneChange != 'NONE':  # len(self._waypoint_buffer) > 0
            newlane, lcopt = None, None
            if doLaneChange == 'LEFT':
                newlane = self._current_waypoint.get_left_lane()
                lcopt = RoadOption.CHANGELANELEFT
            elif doLaneChange == 'RIGHT':
                newlane = self._current_waypoint.get_right_lane()
                lcopt = RoadOption.CHANGELANERIGHT

            if newlane:
                print('newlane wp: lane type = %d, lane_id = %d'
                      % (newlane.lane_type, newlane.lane_id))
            else:
                print('newlane none')
            print('curlane wp: lane type = %d, lane_id = %d'
                  % (self._current_waypoint.lane_type, self._current_waypoint.lane_id))

            if newlane and newlane.lane_type == carla.LaneType.Driving \
                    and sign(newlane.lane_id) == sign(self._current_waypoint.lane_id):
                print('Initiating lane change')

                self._waypoints_queue.clear()
                self._waypoints_queue.append((newlane, lcopt))
                self.get_next_turn_options(need_report_new_road=False)
            else:
                print('Lane change not available')

        if self.manual_brake or self.parked:  # or not self._waypoints_queue:
            control = carla.VehicleControl()

            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        # purge the queue of obsolete waypoints
        max_index = -1
        for i, (waypoint, _) in enumerate(self._waypoints_queue):  
            if waypoint.transform.location.distance(vehicle_transform.location) < self._min_distance:
                max_index = i
            else:
                break
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoints_queue.popleft()


        if not self._waypoints_queue and self.pathSelection:
            print('Reached end of waypoints, taking path selection')
            self._waypoints_queue.append(self.pathSelection)
            self.nextOptions = None
            self.pathSelection = None
            self.get_next_turn_options()

        # target waypoint
        if self._waypoints_queue:
            self.target_waypoint, self._target_road_option = self._waypoints_queue[0]
            # move using PID controllers
            control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)
        else:
            self.target_waypoint, self._target_road_option = self._current_waypoint, RoadOption.LANEFOLLOW
            control = carla.VehicleControl()
            control.brake = 1.0

        if debug:
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], 1.0)

        return control

    def done(self):
        """
        Returns whether or not the planner has finished

        :return: boolean
        """
        return len(self._waypoints_queue) == 0 and len(self._waypoints_queue) == 0

    def set_brake(self, val):
        self.manual_brake = val

    def set_parked(self, val):
        if val == -1:
            self.parked = not self.parked
        else:
            self.parked = val

    def clear_wps(self):
        self._waypoints_queue.clear()
        self.nextOptions = None  # force recalculate

        self.get_next_turn_options()

    def speed_adj(self, val):
        self._target_speed += val

    def select_path_option(self, pathOption): 
        if self.nextOptions and pathOption < len(self.nextOptions):
            self.pathSelection = self.nextOptions[pathOption]
        else:
            print('selected path index %d out of bounds (num options: %d)' % (pathOption, len(self.nextOptions) if self.nextOptions else 0))

    def do_uturn(self):
        cur_wp = self._map.get_waypoint(self._vehicle.get_location())

        scanwp = cur_wp
        while 1:
            if scanwp.lane_id * cur_wp.lane_id > 0:  # if get in bidirectional lane have to go right instead
                scanwp = scanwp.get_left_lane()
            else:
                scanwp = scanwp.get_right_lane()
            if not scanwp: break

            if scanwp.lane_id * cur_wp.lane_id < 0 and scanwp.lane_type == carla.LaneType.Driving:
                print('u-turning to lane %d, new location: (%.2f, %.2f, %.2f)' % (
                scanwp.lane_id, scanwp.transform.location.x, scanwp.transform.location.y, scanwp.transform.location.z))

                self._waypoints_queue.clear()
                self._waypoints_queue.append((scanwp, RoadOption.LANEFOLLOW))
                self.nextOptions = None  # force it to refresh options
                self.get_next_turn_options(need_report_new_road=False)

                break


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT

