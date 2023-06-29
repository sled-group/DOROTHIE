#!/usr/bin/python3

# ==============================================================================
# -- COPYRIGHT -----------------------------------------------------------------
# ==============================================================================

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# Copyright (c) 2022 SLED Lab, EECS, University of Michigan.
# authors: Ben VanDerPloeg (bensvdp@umich.edu),
#          Martin Ziqiao Ma (marstin@umich.edu),
#          Owen Yidong Huang (owenhji@umich.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from multiprocessing import Process
import glob
import os
import imageio
import json
import sys
import math
import csv
import numpy as np
import cv2
# from queue import Queue

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time

from queue import Queue
import argparse





def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)

def dump_images(args,root):
    print(root)
    config_filename = os.path.join(root,"config.json")
    with open(config_filename,'r') as f:
        simconfig=json.load(f)
    town_id = simconfig["environment"]['map']
    recorded_fps = args.fps
    recorder_filename = os.path.join(root,"carlalog.log")
    ad_wizard_logname = os.path.join(root,"ad_wizardlog.csv")

    sensors = ['rgb']  # rgb, depth, semantic_segmentation, instance_segmentation
    if not os.path.exists(os.path.join(root,"out")):
            os.mkdir(os.path.join(root,"out"))
    for sensor in sensors:
        if not os.path.exists(os.path.join(root,"out", sensor)):
            os.mkdir(os.path.join(root,"out", sensor))

    logfn = os.path.join(root,"out",'trajectory.csv')
    if not args.cam_only:
        with open(logfn, 'w') as f:
            f.write('frame, x, y, yaw, junction_id, road_id, lane_id\n')

    ego_vehicle = None
    cam_list = {}
    with open(ad_wizard_logname, newline='') as csvfile:
            reader = csv.reader(csvfile,  delimiter=',',
                     quoting=csv.QUOTE_ALL, skipinitialspace=True)
            ad_wizard_log = []
            for row in reader:
                ad_wizard_log.append(row)
    co_wizard_start_frame=simconfig["start_frame"]
    end_frame=simconfig["end_frame"]
    print(co_wizard_start_frame)
    try:

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        # set the time factor for the replayer
        client.set_replayer_time_factor(args.X)
        # args.fps=args.fps*3
        # set to ignore the hero vehicles or not
        client.set_replayer_ignore_hero(args.ignore_hero)

        # Load the desired map
        client.load_world(town_id)

        # Set world
        # since = time.time()
        world = client.get_world()
        map = world.get_map()
        settings = client.get_world().get_settings()
        if args.sync:
            settings.synchronous_mode = True
            args.use_variable_step = False
        if not args.use_variable_step:
            settings.fixed_delta_seconds = 1 / recorded_fps  # 0.033
            settings.substepping = True
            settings.max_substep_delta_time = 1/(recorded_fps*16)
            settings.max_substeps = 16
        client.get_world().apply_settings(settings)

        # Reload map keeping the world settings
        client.reload_world(False)

        # print the session
        if not args.cam_only:
           with open(os.path.join(root,"out",'log_info.txt' ), 'a') as f:
               f.write(client.show_recorder_file_info(recorder_filename, True))

        # replay the session
        print(client.replay_file(recorder_filename, args.start, args.duration, args.camera))
        if args.sync:
            world.tick()
            world.tick()
            world.tick()
        else:
            world.wait_for_tick()

        # Store the ID from the simulation or query the recording to find out
        if args.agent_id == 0:
            hero_list = [a for a in world.get_actors()
                         if 'role_name' in a.attributes and a.attributes['role_name'] == 'hero']
            if len(hero_list) != 1:
                print('ERROR: Failed to find the co_wizard agent from the logfile!')
            # print(hero_list)
            ego_vehicle = hero_list[0]
        print(args.fps)
        def image_dump(image,frame_id):
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            imageio.imwrite(
                os.path.join(root,"out","rgb",str(image.frame)+".png"),
                        img
                    )
            print(image.frame,frame_id)

        def image_parse(image):
            p = Process(target=image_dump, args=(image,frame_id,))
            p.start()
            # p.join()

        ego_vehicle = world.get_actor(ego_vehicle.id)
        print('Successfully detected co_wizard agent, id = %d' % ego_vehicle.id)
        world.tick()
        # Attach the camera to co_wizard agent
        frame_id = 0
        for i in range(len(sensors)):
            sensor=sensors[i]
            cam_bp = world.get_blueprint_library().find('sensor.camera.%s' % sensor)
            cam_location = carla.Location(1.6,0,1.7)
            cam_rotation = carla.Rotation(0, 0, 0)
            cam_transform = carla.Transform(cam_location, cam_rotation)

            cam_bp.set_attribute("image_size_x", str(args.resolution[0]))  # 1920
            cam_bp.set_attribute("image_size_y", str(args.resolution[1]))  # 1080
            cam_bp.set_attribute("fov", str(105))

            cam_list[sensor]=(world.spawn_actor(cam_bp, cam_transform,
                                              attach_to=ego_vehicle,
                                              attachment_type=carla.AttachmentType.Rigid))
                                  
            cam_list[sensor].listen(image_parse)
        ad_wizard_now_line=1
        while frame_id*args.X<(end_frame-co_wizard_start_frame+300):
            if args.sync:
                world.tick()
            else:
                if not world.wait_for_tick():
                    continue
            while (ad_wizard_now_line <len(ad_wizard_log))and(int(ad_wizard_log[ad_wizard_now_line][1])-co_wizard_start_frame<frame_id*args.X):
                log=ad_wizard_log[ad_wizard_now_line]
                cur_weather = world.get_weather()
                if log[2]=='set_weather':
                    pname=log[3]
                    newval=int(log[4])
                    setattr(cur_weather, pname, newval)
                    world.set_weather(cur_weather)

                ad_wizard_now_line+=1

            transform = ego_vehicle.get_transform()
            wp = map.get_waypoint(transform.location)
            if not args.cam_only and frame_id % (recorded_fps // args.fps) == 0:
                with open(logfn, 'a') as f:
                    f.write('%d, %.6f, %.6f, %.6f, %d, %d, %d\n'
                            % ( world.get_snapshot().frame,
                               transform.location.x,
                               transform.location.y,
                               transform.rotation.yaw,
                               wp.get_junction().id if wp.is_junction else -1 , wp.road_id, wp.lane_id))
            
            frame_id += 1
        time.sleep(20)
        

    finally:
        # --------------
        # Destroy actors
        # --------------
        if ego_vehicle is not None:
            for ego_cam in cam_list:
                if cam_list[ego_cam] is not None:
                    cam_list[ego_cam].stop()
                    cam_list[ego_cam].destroy()
            ego_vehicle.destroy()
        # client.stop_recorder()
        client.stop_replayer(True)
        # client.reload_world()
        print('\nNothing to be done.')


def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
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
        '-s', '--start',
        metavar='S',
        default=0.0,
        type=float,
        help='starting time (default: 0.0)')
    argparser.add_argument(
        '-d', '--duration',
        metavar='D',
        default=0.0,
        type=float,
        help='duration (default: 0.0)')
    argparser.add_argument(
        '-a', '--agent_id',
        default=0,
        help='co_wizard agent id')
    argparser.add_argument(
        '-r', '--root',
        default="/data/owenh/et_dataset_temp",
        type=str,
        help='root file name')
    argparser.add_argument(
        '-c', '--camera',
        metavar='C',
        default=0,
        type=int,
        help='camera follows an actor (ex: 82)')
    argparser.add_argument(
        '-x', '--time-factor',
        metavar='X',
        default=3.0,
        type=float,
        help='How many frames an image is taken (default is 3.0, which is 10 fps)')
    argparser.add_argument(
        '-i', '--ignore-hero',
        action='store_true',
        help='ignore hero vehicles')
    argparser.add_argument(
        '--spawn-sensors',
        action='store_true',
        help='spawn sensors in the replayed world')
    argparser.add_argument(
        '--sync',
        action='store_true',
        dest='sync',
        help='Enable CARLA synchronous mode. This seems to cause issues with Dorothy.')
    argparser.add_argument(
        '--variable-step',
        action='store_true',
        dest='use_variable_step',
        help='Enable CARLA variable step mode. Default is fixed step mode.')
    argparser.add_argument(
        '--fps',
        help='The recoding fps.',
        dest='fps',
        default=30,
        type=int)
    argparser.add_argument(
        '--sample_interval',
        help='Set bounding box sample interval, default is 2 second',
        dest='sample_interval',
        default=2,
        type=int)
    argparser.add_argument(
        '--resolution',
        help='Set image resolution.',
        dest='resolution',
        default=(480, 270),
        type=tuple)
    argparser.add_argument(
        '--camera-only',
        action='store_true',
        dest='cam_only',
        help='Record only camera log.')
    args = argparser.parse_args()
    for session in os.listdir(args.root):
        dump_images(args=args,root=os.path.join(args.root,session))
        time.sleep(10000)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
