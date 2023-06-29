#!/usr/bin/python3

# ==============================================================================
# -- COPYRIGHT -----------------------------------------------------------------
# ==============================================================================

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
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
This is the main interface of the co_wizard operator.
"""

# ==============================================================================
# -- IMPORTS -------------------------------------------------------------------
# ==============================================================================

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref
import threading
import json
from queue import Queue
import socket
import time

import co_wizard_map as co_wizard_map
from config_gen import config_gen, storyboard_gen

fmod = math.fmod

# from scipy.spatial.transform import Rotation as ScipyRotation
SUBGOAL_DISTANCE = 4
DELTA_SPEED = 5
WAIT_TIME=(5,10)
# for blurring view when ad_wizard
from scipy.ndimage import gaussian_filter
BLUR_ONSET_TIME = 8

SPEECH_EN = True

if SPEECH_EN:
    import co_wizard_speech_server as co_wizard_speech_server

try:
    import pygame
    from pygame.locals import *
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

sys.path.insert(0, '../util')
import pedctrl


# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
from co_wizard_agent import Co_WizardAgent

if not os.path.exists('co_wizardlogs'):
    os.mkdir('co_wizardlogs')
if not os.path.exists('carlalogs'):
    os.mkdir('carlalogs')

episode_timestamp = time.time()
os.mkdir('co_wizardlogs/log_%d' % episode_timestamp)
logfn = 'co_wizardlogs/log_%d/co_wizardlog_%d.csv' % (episode_timestamp, episode_timestamp)
with open(logfn, 'a') as f:
    f.write('systime, frame, from, to, action_type, action_name, args\n')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

WEATHER_PARAM_NAMES = ('cloudiness',
                       'precipitation',
                       'precipitation deposits',
                       'wind intensity',
                       'sun azimuth angle',
                       'sun altitude angle',
                       'fog density',
                       'fog distance',
                       'fog falloff',
                       'wetness')

# Only ones with headlights. 'vehicle.volkswagen.t2' is also an option but has reduced visibility
acceptable_vehicle_filters = ['vehicle.audi.tt',
                              'vehicle.chevrolet.*',
                              'vehicle.audi.etron',
                              'vehicle.lincoln.*',
                              'vehicle.mustang.*',
                              'vehicle.tesla.model3']


def find_weather_presets():
    """
    Method to find weather presets
    """
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """
    Method to get actor display name
    """
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    """
    Class representing the surrounding environment.
    """

    def __init__(self, carla_world, hud, args, comms=None, simconfig=None, map_input_queue=None, carlaclient=None):
        """
        Constructor method
        """
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.curWeather = self._weather_presets[0]
        self._gamma = args.gamma
        self.comms = comms
        self.simconfig = {} if simconfig is None else simconfig
        self.map_input_queue = map_input_queue
        self.carlaclient = carlaclient

        self.restart(args)

        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.userinput = ''
        self.parked = True

        self.last_args = args
        if not hasattr(self, 'prop_actors') : self.prop_actors = []

        # lpqueue relays keyboard commands to the local planner
        self.lpqueue = Queue()
        self.lpqueue.put(('parked', True))

    def restart(self, args=None):
        """
        Restart the world
        """
        if not args:
            args = self.last_args
# 
        # Set the seed if requested by user
        if args.seed is not None:
            random.seed(args.seed)

        bpl = self.world.get_blueprint_library()
        blueprint = bpl.find(self.simconfig['vehicle'])

        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player.
        print("Spawning the player")
        spawn_point = carla.Location(x=0, y=0, z=0)
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            spawn_point = carla.Transform(location=carla.Location(*self.simconfig['departure']['location']),
                                          rotation=carla.Rotation(*self.simconfig['departure']['rotation']))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        with open(logfn, 'a') as f:
            ss = self.world.get_snapshot()
            f.write('%.3f, %d, "system", "agent", "status", "spawn", %d\n'
                    % (time.time(), ss.frame, self.player.id))

        if 'npc_count' in self.simconfig:
            print('Spawning NPCs')
            pedctrl.createVehicles(self.carlaclient, self.simconfig['npc_count'], tm_port=args.tm_port)

        print('Spawning props')
        self.prop_actors = []
        self.prop_actors_map = {}

        for inst in self.simconfig['assets']:
            if inst['type'].startswith('null.'):
                # asset with no physical form
                self.prop_actors.append(None)
                continue

            bp = bpl.find(inst['type'])
            tf = carla.Transform(location=carla.Location(*inst['location']),
                                 rotation=carla.Rotation(*inst['rotation']))
            ac = self.world.spawn_actor(bp, tf)

            if inst['type'].startswith('static'):
                tf1 = tf
                tf1.rotation.yaw += 180
                tf1.location.x -= 0.5
                ac_back = self.world.spawn_actor(bp, tf1)

            self.prop_actors.append(ac)
            self.prop_actors_map[inst['type']] = ac

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.set_sensor( notify=False)
        actor_type = get_actor_display_name(self.player)

    def next_weather(self, reverse=False):
        """
        Get next weather setting
        """
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        self.curWeather = self._weather_presets[self._weather_index]
        # self.hud.notification('Weather: %s' % self.curWeather[1])
        print(self.curWeather)
        self.player.get_world().set_weather(self.curWeather[0])

    def tick(self, clock):
        """
        Method for every tick
        """
        self.hud.tick(self, clock)

    def render(self, display):
        """
        Render world
        """
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """
        Destroy sensors
        """
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor_top.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """
        Destroys all actors
        """
        actors = [
            self.camera_manager.sensor,
            self.camera_manager.sensor_top,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    def __init__(self, world, map_pygame_event_queue):

        self.world = world
        self.isUserTyping = False
        self.map_pygame_event_queue = map_pygame_event_queue

        self.sys_reset_press_time = None


        if SPEECH_EN:
            self.sr_control = Queue()
            self.speech_thread = threading.Thread(target=co_wizard_speech_server.run,
                                                  args=(self.sr_control,
                                                        self.world.comms.speech_inbox,
                                                        self.world.comms.speech_outbox,
                                                        episode_timestamp),
                                                  daemon=True)
            self.speech_thread.start()

    def parse_events(self):

        ss = self.world.world.get_snapshot()
        location = self.world.player.get_transform().location
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return True

            elif event.type == pygame.KEYDOWN:
                if event.key == K_t:
                    self.sr_control.put('start')
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "dorothy", "communication", "start", "NONE"\n'
                                % (time.time(), ss.frame))

                elif event.key == K_BACKSLASH:
                    self.sys_reset_press_time = time.time()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "system", "status", "termination", "NONE"\n'
                                % (time.time(), ss.frame))
                    return True

                # Left lane change
                elif event.key == K_a:

                    self.world.lpqueue.put(('laneChangeReq', 'LEFT'))
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "agent", "decision", "lane_change_left"\n'
                                % (time.time(), ss.frame))

                # Right lane change
                elif event.key == K_d:

                    self.world.lpqueue.put(('laneChangeReq', 'RIGHT'))
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "agent", "decision", "lane_change_right"\n'
                                % (time.time(), ss.frame))

                # Release manual brake
                elif event.key == K_r:
                    print('Manual brake release')
                    self.world.lpqueue.put(('brake', False))
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "agent", "decision", "manual_brake_release"\n'
                                % (time.time(), ss.frame))

                # Engage manual brake
                elif event.key == K_s:
                    print('Manual brake')
                    self.world.lpqueue.put(('brake', True))
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "agent", "decision", "manual_brake_engage"\n'
                                % (time.time(), ss.frame))

                # Clear waypoints and make new plan
                elif event.key == K_c:
                    print('Clear waypoints')
                    self.world.lpqueue.put(('clear_wps', None))
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "agent", "plan", "waypoint_clear"'
                                % (time.time(), ss.frame))

                # Increase speed by 5 km/h
                elif event.key == K_UP:
                    self.world.lpqueue.put(('speedAdj', DELTA_SPEED))
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "agent", "decision", "speed_up", %d\n'
                                % (time.time(), ss.frame, DELTA_SPEED))

                # Decrease speed by 5 km/h
                elif event.key == K_DOWN:
                    self.world.lpqueue.put(('speedAdj', -DELTA_SPEED))
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "agent", "decision", "speed_down", -%d\n'
                                % (time.time(), ss.frame, DELTA_SPEED))

                # Enter/Exit "parked" state
                elif event.key == K_f:
                    self.world.parked = not self.world.parked
                    self.world.lpqueue.put(('parked', self.world.parked))

                    # Park
                    if self.world.parked:
                        self.world.comms.co_wizard_outbox.put('dorothy:co_wizard:parkon')
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "co_wizard", "agent", "decision", "park_on"\n'
                                    % (time.time(), ss.frame))

                    # Restart
                    else:
                        self.world.comms.co_wizard_outbox.put('dorothy:co_wizard:parkoff')
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "co_wizard", "agent", "decision", "park_off", "none"\n'
                                    % (time.time(), ss.frame))

                # Make a U-turn
                elif event.key == K_u:
                    self.world.lpqueue.put(('uturn', None))
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "agent", "decision", "uturn", "none"\n'
                                % (time.time(), ss.frame))

                # Speech control
                elif event.key == K_t:
                    self.sr_control.put('stop')

                # Turn on/off the light
                elif event.key == K_l:
                    old_state = self.world.player.get_light_state()

                    # Light on
                    if old_state & carla.VehicleLightState.LowBeam:
                        self.world.player.set_light_state(carla.VehicleLightState.NONE)
                        print('lights off')
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "co_wizard", "agent", "decision", "headlights_off"\n'
                                    % (time.time(), ss.frame))

                    # Light off
                    else:
                        self.world.player.set_light_state(carla.VehicleLightState.LowBeam)
                        print('lights on')
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "co_wizard", "agent", "decision", "headlights_on"\n'
                                    % (time.time(), ss.frame))

                # Reset session
                elif event.key == K_BACKSLASH:
                    if time.time() - self.sys_reset_press_time > 1:
                        print('Resetting world')
                        self.world.player.set_transform(self.world.map.get_spawn_points()[0])
                        self.world.lpqueue.put(('clear_wps', None))
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "co_wizard", "environment", "none", "world_reset"\n'
                                    % (time.time(), ss.frame))
                        # self.world.restart()

                # Specify a new plan
                elif event.key == K_p:
                    self.world.map_input_queue.put(('toggle_plan_mode', None))

                # Undo previous plan annotation entry
                elif event.key == K_z:
                    self.map_pygame_event_queue.put(event)

            elif event.type == pygame.MOUSEBUTTONUP:
                self.map_pygame_event_queue.put(event)  # the map only cares about mousebuttonup so only send those

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """
    Class for HUD text
    """

    def __init__(self, width, height):
        """
        Constructor method
        """
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """
        Get information from the world at every tick
        """
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """
        HUD method for every tick
        """
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        self._info_text += ['User input: ' + world.userinput]

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """
        Toggle info on or off
        """
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """
        Notification text
        """
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """
        Error text
        """
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """
        Render for HUD class
        """
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n') if __doc__ else ['hello']
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- Camera Manager ------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud, gamma_correction):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transform = (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid)
        self._camera_transform_top = (carla.Transform(
                carla.Location( z=70),carla.Rotation(pitch=-90)), attachment.Rigid)
        self.sensors = [['sensor.camera.rgb', cc.Raw, 'Camera RGB'],['sensor.camera.rgb', cc.Raw, 'Camera RGB']]
        
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        blp = bp_library.find(self.sensors[0][0])
        blp.set_attribute('image_size_x', str(hud.dim[0]))
        blp.set_attribute('image_size_y', str(hud.dim[1]))
        if blp.has_attribute('gamma'):
            blp.set_attribute('gamma', str(gamma_correction))
        self.sensors[0].append(blp)
        blp = bp_library.find(self.sensors[1][0])
        blp.set_attribute('image_size_x', str(720))
        blp.set_attribute('image_size_y', str(720))
        self.sensors[1].append(blp)
        
        
    def set_sensor(self,  notify=True, force_respawn=False):
        """Set a sensor"""
        if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
        self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[0][-1],
                self._camera_transform[0],
                attach_to=self._parent,
                attachment_type=self._camera_transform[1])
        self.sensor_top = self._parent.get_world().spawn_actor(
                self.sensors[1][-1],
                self._camera_transform_top[0],
                attach_to=self._parent,
                attachment_type=self._camera_transform_top[1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        
    def get_camera_ids(self):
        return self.sensor.id , self.sensor_top.id 
    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.sensors[0][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.image_np = array
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- Communication Manager -----------------------------------------------------
# ==============================================================================

class CommsManager:
    """
    communication thread with ad_wizard and dorothy
    """
    def __init__(self, args, hud):
        """
        start the communication thread
        """
        self.frame = -1

        def tcp_communication(port, inbox, outbox):
            """
            The communication thread with ad_wizard or dorothy
            """
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('0.0.0.0', port))
                sock.listen(1)

                while 1:
                    connection, client_addr = sock.accept()
                    connection.setblocking(False)

                    print('New connection from %s on port %d' % (str(client_addr), port))

                    textIn = ''
                    nextKeepalive = time.time()

                    try:
                        while 1:
                            try:
                                textIn += str(connection.recv(1024), "utf-8", errors='ignore')
                                if '\n' in textIn:
                                    endpos = textIn.index('\n')
                                    inbox.put(textIn[:endpos].strip())
                                    textIn = textIn[endpos+1:]

                            except BlockingIOError:
                                # ignore if no input available
                                pass

                            if not outbox.empty():
                                msg = outbox.get()
                                print('sending %s on port %d' % (msg, port))
                                connection.sendall(bytes(msg.strip() + '\n', 'utf-8', errors='ignore'))

                            if time.time() > nextKeepalive:
                                connection.sendall(b'keepalive\n')
                                nextKeepalive += 1.0

                            time.sleep(0.05)
                    except ConnectionError:
                        pass
                    finally:
                        connection.close()
                        print('Connection from %s on port %d closed' % (str(client_addr), port))

        def manage(dorothy_inbox, dorothy_outbox,
                   ad_wizard_inbox, ad_wizard_outbox,
                   co_wizard_inbox, co_wizard_outbox,
                   speech_inbox, speech_outbox,hud):
            '''
            manage the inbox and outbox
            '''
            while True:
                if not ad_wizard_inbox.empty():
                    msg = ad_wizard_inbox.get()
                    print('from ad_wizard: %s' % msg)

                    if msg == 'dorothy:ad_wizard:action:trigger':
                        print(1)
                        dorothy_outbox.put(msg)
                        co_wizard_inbox.put(msg)
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "dorothy", "ad_wizard", "trigger", "once", "once"\n'
                                    % (time.time(), self.frame))

                    if msg == 'dorothy:ad_wizard:action:delete':
                        print(1)
                        dorothy_outbox.put(msg)
                        co_wizard_inbox.put(msg)
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "dorothy", "ad_wizard", "delete", "once", "once"\n'
                                    % (time.time(), self.frame))

                    if msg == 'dorothy:ad_wizard:action:change':
                        print(1)
                        dorothy_outbox.put(msg)
                        co_wizard_inbox.put(msg)
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "dorothy", "ad_wizard", "change", "once", "once"\n'
                                    % (time.time(), self.frame))

                    elif msg.startswith('dorothy:'):
                        dorothy_outbox.put(msg)
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "ad_wizard", "dorothy", "communication", "end", "%s"\n'
                                    % (time.time(), self.frame, msg))

                    # We are in the co_wizard executable, so messages go to inbox (vs outbox for dorothy and ad_wizard)
                    elif msg.startswith('co_wizard:'):
                        co_wizard_inbox.put(msg)
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "ad_wizard", "co_wizard", "communication", "end", "%s"\n'
                                    % (time.time(), self.frame, msg))
                    
                    else:
                        print('NOTE: ill-formatted message from ad_wizard: %s' % msg)

                if not dorothy_inbox.empty():
                    msg = dorothy_inbox.get()
                    print('from dorothy: %s' % msg)

                    # Utterance from dorothy for co_wizard. put in speech queue
                    if msg.startswith('co_wizard:dorothy:message:'):
                        cntent = msg[len('co_wizard:dorothy:message:'):]
                        print('dorothy said: %s' % cntent)
                        speech_inbox.put(cntent)
                        raw_text = msg.split(':')[3]
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "dorothy", "co_wizard", "communication", "end", "%s", "%s", "%s"\n'
                                    % (time.time(), self.frame, raw_text, hud.server_fps, hud._server_clock.get_fps()))

                    # All other co_wizard-bound messages go to regular (non-speech) co_wizard inbox
                    elif msg.startswith('co_wizard:'):
                        co_wizard_inbox.put(msg)
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "dorothy", "co_wizard", "communication", "end", "%s"\n'
                                    % (time.time(), self.frame, msg))

                    elif msg.startswith('ad_wizard:'):
                        ad_wizard_outbox.put(msg)
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "dorothy", "ad_wizard", "communication", "end", "%s"\n'
                                    % (time.time(), self.frame, msg))

                    else:
                        print('NOTE: ill-formatted message from dorothy: %s' % msg)

                if not co_wizard_outbox.empty():
                    msg = co_wizard_outbox.get()
                    if msg.startswith('dorothy:'):
                        dorothy_outbox.put(msg+'\n')
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "system", "dorothy", "system", "%s"\n'
                                    % (time.time(), self.frame, msg))

                    elif msg.startswith('ad_wizard:'):
                        ad_wizard_outbox.put(msg+'\n')
                        print(msg)
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "system", "ad_wizard", "system", "%s"\n'
                                    % (time.time(), self.frame, msg))

                    else:
                        print('NOTE: ill-formatted message from co_wizard: %s' % msg)

                # Speech utterances for dorothy
                if not speech_outbox.empty():
                    msg = speech_outbox.get()
                    dorothy_outbox.put('dorothy:co_wizard:message:' + msg.strip() + '\n')
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "dorothy", "communication", "end", "%s"\n'
                                % (time.time(), self.frame, msg.strip()))

                time.sleep(0.1)

        dorothy_inbox, dorothy_outbox, ad_wizard_inbox, ad_wizard_outbox = Queue(), Queue(), Queue(), Queue()
        self.dorothy_tcp_thread = threading.Thread(target=tcp_communication,
                                                   args=(args.dorothyport, dorothy_inbox, dorothy_outbox), daemon=True)
        self.dorothy_tcp_thread.start()
        self.ad_wizard_tcp_thread = threading.Thread(target=tcp_communication,
                                                 args=(args.ad_wizardport, ad_wizard_inbox, ad_wizard_outbox), daemon=True)
        self.ad_wizard_tcp_thread.start()

        self.co_wizard_inbox, self.co_wizard_outbox, self.speech_inbox, self.speech_outbox = Queue(), Queue(), Queue(), Queue()
        self.manage_thread = threading.Thread(target=manage,
                                              args=(dorothy_inbox, dorothy_outbox, ad_wizard_inbox, ad_wizard_outbox,
                                                    self.co_wizard_inbox, self.co_wizard_outbox,
                                                    self.speech_inbox, self.speech_outbox,hud), daemon=True)
        self.manage_thread.start()

    def set_frame(self, frame_new):
        self.frame = frame_new


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    """
    Main game loop for agent
    """

    pygame.init()
    pygame.font.init()
    world = None
    client = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(101.0)

        if args.metaconfig:
            if not os.path.exists('configs'):
                os.mkdir('configs')
            args.config = 'co_wizardlogs/log_%d/autoconfig_%d.json' % (episode_timestamp, episode_timestamp)
            config_gen(args.metaconfig, args.config)

        if not os.path.exists(args.config):
            print('ERROR: config file %s not found' % args.config)
            sys.exit(1)
        with open(args.config) as f:
            simconfig = json.load(f)

        if args.sb:
            with open(args.sb) as f:
                storyboard = json.load(f)
        elif args.tpl:
            sb_file = 'co_wizardlogs/log_%d/storyboard_autogen_%d.json' % (episode_timestamp, episode_timestamp)
            storyboard = storyboard_gen(args.tpl, sb_file, simconfig)
        else:
            print('ERROR: Must either specify storyboard or storyboard template.')
            sys.exit(1)

        with open('../common/asset_metadata.json') as f:
            asset_md = json.load(f)

        astname_to_id = {}
        for astid, astdata in asset_md.items():
            astname_to_id[astdata['name']] = astid

        subgoal_assets = []
        for sg in storyboard['subgoals']:
            assert sg['destination'] in astname_to_id, 'ERROR: unknown storyboard asset %s' % sg['destination']
            targetid = astname_to_id[sg['destination']]

            sgast = None
            for ast in simconfig['assets']:
                if ast['type'] == targetid:
                    sgast = ast
                    break
            assert sgast, 'ERROR: no matching asset found for "%s" in storyboard (expected asset id: %s)' \
                          % (sg['destination'], targetid)
            if 'change_destination' in sg:
                targetid = astname_to_id[sg['change_destination']]
                for ast in simconfig['assets']:
                    if ast['type'] == targetid:
                        sgast['change_destination']=ast
            sgast['change']=sg['change']
            sgast["arrived"] = False
            sgast["trigger"] = sg['trigger'] if 'trigger' in sg else False
            sgast["delete"] = sg['delete'] if 'delete' in sg else False
            sgast["after"] = sg['after'] if 'after' in sg else []
            subgoal_assets.append(sgast)
        print('subgoal assets: ' + str(subgoal_assets))
        subgoal_cnt = 0

        map_metadata_fname = '../common/map_metadata/%s.json' % simconfig['map']
        if not os.path.exists(map_metadata_fname):
            print('ERROR: map metadata file %s not found' % map_metadata_fname)
            sys.exit(1)
        with open(map_metadata_fname) as f:
            simconfig['map_metadata'] = json.load(f)

        client.load_world(simconfig['map'])
        if 'weather' in simconfig:
            weather = client.get_world().get_weather()
            for par, pval in simconfig['weather'].items():
                setattr(weather, par, pval)
            client.get_world().set_weather(weather)

        settings = client.get_world().get_settings()
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(1)
        traffic_manager.set_random_device_seed(args.seed)

        if args.sync:
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            args.use_variable_step = False

        if not args.use_variable_step:
            settings.fixed_delta_seconds = 1 / args.fps  # 0.033
            settings.substepping = True
            # Smallest substep possible, assuming 16 substeps max (with small margin) # 0.005
            settings.max_substep_delta_time = 1/(args.fps*16)  # 0.002
            settings.max_substeps = 16
            assert settings.fixed_delta_seconds <= settings.max_substep_delta_time * settings.max_substeps

        client.get_world().apply_settings(settings)
        print('Sync, Variable-Step, FPS:', args.sync, args.use_variable_step, args.fps)

        mapimg = pygame.image.load('../common/map_images/%s.jpg' % simconfig['map'])
        MAPWIDTH, MAPHEIGHT = mapimg.get_rect().size
        del mapimg

        display = pygame.display.set_mode(
            (args.width + MAPWIDTH, max(args.height, MAPHEIGHT)),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        # Just for the 3d view, gets blitted onto the main display
        display_3d_view = pygame.Surface((args.width, args.height))
        pygame.display.set_caption("Co_Wizard")

        map_output_queue, map_input_queue, map_frames_queue, map_pygame_event_queue = Queue(), Queue(), Queue(maxsize=1), Queue()
        map_thread = threading.Thread(target=co_wizard_map.run_co_wizard_map,
                                      args=(args, simconfig, storyboard, map_output_queue,
                                            map_input_queue, map_frames_queue, map_pygame_event_queue, logfn),
                                      kwargs={'mouse_x_offset': -args.width}, daemon=True)
        map_thread.start()
        hud = HUD(args.width, args.height)
        cm = CommsManager(args,hud)

        world = World(client.get_world(), hud, args,
                      comms=cm, simconfig=simconfig, map_input_queue=map_input_queue, carlaclient=client)
        ss = world.world.get_snapshot()
        controller = KeyboardControl(world, map_pygame_event_queue)

        # Queue used to relay commands from user interface to agent/planner
        agent = Co_WizardAgent(world.player, map_input_queue=map_input_queue, lpqueue=world.lpqueue)

        bigfont = pygame.font.Font(pygame.font.get_default_font(), 48)

        audio_path = 'co_wizardlogs/log_%d/co_wizard_audio/' % episode_timestamp
        if not os.path.exists(audio_path):
            os.mkdir(audio_path)

        clock = pygame.time.Clock()
        client.start_recorder(args.logfile, additional_data=True)
        with open(logfn, 'a') as f:
            f.write('%.3f, %d, "system", "agent", "status", "start_record", "NONE"\n'
                    % (time.time(), ss.frame))

        # Flag set by incoming commands from ad_wizard
        ad_wizard_blur = False
        blur_start_time = 0
        emergency_stop = False
        waiting=False
        wait_start=0
        wait_time=0
        triggered=False
        deleted=False
        changed=False
        while True:
            while not map_output_queue.empty():
                cmd, pars = map_output_queue.get()
                if cmd == 'chosen_path':
                    world.lpqueue.put(('pathOption', pars))
                else:
                    print('unrecognized map rx command: %s' % cmd)

            
            # deal with input messages 
            
            while not cm.co_wizard_inbox.empty():
                msg = cm.co_wizard_inbox.get().strip()

                if msg == 'co_wizard:ad_wizard:bluron':
                    ad_wizard_blur = True
                    blur_start_time = time.time()

                elif msg == 'dorothy:ad_wizard:action:trigger':
                    triggered=True
                elif msg == 'dorothy:ad_wizard:action:delete':
                    deleted=True
                elif msg == 'dorothy:ad_wizard:action:change':
                    for sg in subgoal_assets:
                        if 'change_destination' in sg:
                            sg['type']=sg['change_destination']['type']
                            sg['location']=sg['change_destination']['location']
                            sg['rotation']=sg['change_destination']['rotation']
                    print(str(subgoal_assets))

                elif msg == 'co_wizard:ad_wizard:bluroff':
                    ad_wizard_blur = False
                    blur_start_time = time.time()

                elif msg.endswith('config_rq'):
                    dest = msg.split(':')[1]
                    print('Config request received, sending')
                    cm.co_wizard_outbox.put(
                        '%s:co_wizard:config:%s' % (dest, json.dumps(simconfig, separators=(',', ':'))))

                elif msg.endswith('sb_rq'):
                    dest = msg.split(':')[1]
                    print('Storyboard request received, sending')
                    cm.co_wizard_outbox.put(
                        '%s:co_wizard:sb:%s' % (dest, json.dumps(storyboard, separators=(',', ':'))))

                elif msg.endswith('cam_rq'):
                    dest = msg.split(':')[1]
                    print('camera id request received, sending')
                    cam_id,cam_top_id=world.camera_manager.get_camera_ids()
                    id_dict={"cam_id":cam_id,"cam_top_id":cam_top_id}
                    cm.co_wizard_outbox.put(
                        '%s:co_wizard:sb:%s' % (dest, json.dumps(id_dict, separators=(',', ':'))))

                elif msg == 'co_wizard:ad_wizard:nogpson':
                    map_input_queue.put(('nogpson', None))

                elif msg == 'co_wizard:ad_wizard:nogpsoff':
                    map_input_queue.put(('nogpsoff', None))

                else:
                    print('UNRECOGNIZED: %s' % msg)

            # detech subgoal
            for cur_subgoal in range(len(subgoal_assets)):
                if subgoal_assets[cur_subgoal]["arrived"]:
                    continue
                if (subgoal_assets[cur_subgoal]['trigger']&(not triggered)):
                    continue
                if (subgoal_assets[cur_subgoal]['delete']&(deleted)):
                    continue
                indepen=True
                for i in subgoal_assets[cur_subgoal]['after']:
                    if not subgoal_assets[i]["arrived"]:
                       indepen=False
                if not indepen:
                    continue
                
                csx, csy, csz = subgoal_assets[cur_subgoal]['location']
                location=carla.Location(csx,csy,csz)
                waypoint=world.map.get_waypoint(location)
                cs=waypoint.transform.location
                vpos = world.player.get_location()
                if (cs.x - vpos.x)**2 + (cs.y - vpos.y)**2 + (cs.z - vpos.z)**2 < SUBGOAL_DISTANCE:
                    print('SUBGOAL ACCOMPLISHED: reached %s' % storyboard['subgoals'][cur_subgoal]['destination'])
                    if storyboard['subgoals'][cur_subgoal]['delete_after_subgoal']:
                        world.prop_actors_map[subgoal_assets[cur_subgoal]['type']].destroy()
                    cm.co_wizard_outbox.put('dorothy:co_wizard:sb_reach:%s' %str(cur_subgoal))
                    subgoal_assets[cur_subgoal]["arrived"]=True
                    subgoal_cnt+=1
                    if subgoal_cnt >= len(subgoal_assets):
                        print('FINAL GOAL REACHED')

            clock.tick_busy_loop(60)
            if controller.parse_events():
                return

            # world.wait_for_tick is only used in asynchronous mode. use world.tick for synchronous
            # As soon as the server is ready continue!
            if args.sync:
                world.world.tick(10.0)
            else:
                if not world.world.wait_for_tick(10.0):
                    continue
                
            # control traffic light
            
            if world.player.is_at_traffic_light():
                traffic_light = world.player.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    if waiting :
                        if (time.time()-wait_start>=wait_time):
                            traffic_light.set_state(carla.TrafficLightState.Green)
                            traffic_light.set_green_time(20.0-wait_time)
                            waiting=False
                    else:
                        waiting=True
                        lights=traffic_light.get_group_traffic_lights()
                        for light in lights:
                            traffic_light.set_state(carla.TrafficLightState.Red)
                            traffic_light.set_red_time(20.0)
                        wait_time=random.randrange(WAIT_TIME[0],WAIT_TIME[1])
                        wait_start=time.time()

            world.tick(clock)
            world.render(display_3d_view)
            cm.set_frame(world.world.get_snapshot().frame)

            bfrac = (time.time() - blur_start_time) / BLUR_ONSET_TIME
            if bfrac < 1 or ad_wizard_blur:
                img_np = pygame.surfarray.array3d(display_3d_view)
                img_np_blurred = gaussian_filter(img_np, (9, 9, 1))

                if bfrac >= 1:  # fully blurred
                    pygame.surfarray.blit_array(display_3d_view, img_np_blurred)
                else:
                    if not ad_wizard_blur: bfrac = 1 - bfrac # check if blur is fading
                    pygame.surfarray.blit_array(display_3d_view, img_np_blurred*bfrac + img_np*(1 - bfrac))

            if world.parked:
                textsurf = bigfont.render('Parked', True, (255, 0, 0))
                display_3d_view.blit(textsurf, (args.width//2, args.height//2))

            display.blit(display_3d_view, (0, 0)) # left-align 3D view
            if not map_frames_queue.empty():
                # print('received map frame')
                display.blit(map_frames_queue.get(), (args.width, 0))

            pygame.display.flip()
            control, hazard = agent.run_step()  # debug=True)
            control.manual_gear_shift = False
            world.player.apply_control(control)
            ss = world.world.get_snapshot()

            if hazard is None:
                if emergency_stop:
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "planner", "agent", "decision", "restart", "none"\n'
                                % (time.time(), ss.frame))
                    emergency_stop = False

            elif not emergency_stop:
                emergency_stop = True
                reason, id = hazard
                with open(logfn, 'a') as f:
                    f.write('%.3f, %d, "planner", "agent", "decision", "stop", "%s:%d"\n'
                            % (time.time(), ss.frame, reason, id))

    finally:
        if client is not None:
            client.stop_recorder()
        if world is not None:
            world.destroy()

        # if lcmInstance and lcmThread.is_alive():
        #     lcmThread.join()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    """
    Main method
    """
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')

    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8010,
        type=int,
        help='Port to communicate with TM (default: 8010)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: 0)',
        default=0,
        type=int)
    argparser.add_argument(
        '-c', '--config',
        help='JSON config file for props in scene (default: props_config.json)',
        default='props_config.json',
        type=str)
    argparser.add_argument(
        '-m', '--metaconfig',
        help='JSON "metaconfig" which will be used to generate a config. '
             'Generated config will be saved to ./autoconfigs. '
             'If specified, --config will be ignored.',
        type=str)
    argparser.add_argument(
        '--sb',
        help='Storyboard to use',
        type=str
    )
    argparser.add_argument(
        '--tpl',
        help='Storyboard template to use',
        type=str
    )
    argparser.add_argument(
        '-l', '--logfile',
        help='CARLA logfile',
        default=os.path.join(os.getcwd(), 'co_wizardlogs/log_%d/carlalog_%d.log' % (episode_timestamp, episode_timestamp)),
        type=str)
    argparser.add_argument(
        '--ad_wizard-port',
        help='Port for ad_wizard TCP connection',
        dest='ad_wizardport',
        default=6790,
        type=int)
    argparser.add_argument(
        '--dorothy-port',
        help='Port for dorothy TCP connection',
        dest='dorothyport',
        default=6791,
        type=int)
    argparser.add_argument(
        '--variable-step',
        action='store_true',
        dest='use_variable_step',
        help='Enable CARLA variable step mode. Default is fixed step mode.')
    argparser.add_argument(
        '--fps',
        help='Set carla fixed-step FPS. Default is 30.',
        dest='fps',
        default=30,
        type=int)
    argparser.add_argument(
        '--sync',
        action='store_true',
        dest='sync',
        help='Enable CARLA synchronous mode. This seems to cause issues with Dorothy.')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
