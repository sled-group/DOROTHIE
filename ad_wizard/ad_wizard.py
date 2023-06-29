#!/usr/bin/python3

# ==============================================================================
# -- COPYRIGHT -----------------------------------------------------------------
# ==============================================================================

# Copyright (c) 2022 SLED Lab, EECS, University of Michigan.
# authors: Martin Ziqiao Ma (marstin@umich.edu),
#          Ben VanDerPloeg (bensvdp@umich.edu),
#          Owen Yidong Huang (owenhji@umich.edu),
#          Elina Eui-In Kim (euiink@umich.edu),
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
import pygame
import threading
from queue import Queue
import socket
import time
import argparse
import json
import sys
import os

from math import sqrt, fmod, floor
import numpy as np
argmin = np.argmin
radians = np.radians
sin = np.sin
cos = np.cos

from ad_wizard_utils import Button, transform2M , mToE,messageClient

# Clean this up
sys.path.insert(0, '../util')
import pedctrl

from avutil import render_map_from_simconfig, line_seg_to_point
from pygame.locals import *

# Show car location on map
EN_EGO = True
DEBUG = False

WEATHER_CLICK_INC = 2
WEATHER_DRAG_FACTOR = 0.2
opt_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]

headings = ['east', 'southeast', 'south', 'southwest', 'west', 'northwest', 'north', 'northeast']

# NOTE: the Z coordinate of these default locations is interpreted as the vertical offset from the ground

props = {
    'tree': ('static.prop.tree', carla.Transform(location=carla.Location(0, 0, 0),
                                                 rotation=carla.Rotation(45, 135, 45))),
    'constructioncone': ('static.prop.constructioncone', carla.Transform(location=carla.Location(0, 0, 0.01),
                                                                         rotation=carla.Rotation(0, 0, 0))),
    'dumpster': ('static.prop.container', carla.Transform(location=carla.Location(0, 0, 0.01),
                                                          rotation=carla.Rotation(0, 0, 0))),
    'streetbarrier': ('static.prop.streetbarrier', carla.Transform(location=carla.Location(0, 0, 0.01),
                                                                   rotation=carla.Rotation(0, 0, 0))),
    'trafficcone': ('static.prop.trafficcone01', carla.Transform(location=carla.Location(0, 0, 0.01),
                                                                 rotation=carla.Rotation(0, 0, 0))),
    'trafficsign': ('static.prop.trafficwarning', carla.Transform(location=carla.Location(0, 0, 0.01),
                                                                  rotation=carla.Rotation(0, 0, 0))),
}

WEATHER_PARAM_NAMES = ('cloudiness',
                       'precipitation',
                       'precipitation_deposits',
                       'wind_intensity',
                       'sun_azimuth_angle',
                       'sun_altitude_angle',
                       'fog_density',
                       'fog_distance',
                       'fog_falloff',
                       'wetness')


def worldToPixel(loc,mapscale,xint,yint):
    if isinstance(loc, tuple) or isinstance(loc, list):
        return int(loc[0] * mapscale + xint), int(loc[1] * mapscale + yint)
    return int(loc.x * mapscale + xint), int(loc.y * mapscale + yint)


def pixelToWorld(p,mapscale,xint,yint):
    return (p[0] - xint) / mapscale, (p[1] - yint) / mapscale


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


class Ad_wizard:
    def __init__(self,client,world,screen,myfont,loaded_thumbnails,roads,bg,outbox,logfn,weather_buttons) -> None:
        self.world=world
        self.car = None
        self.client=client
        if EN_EGO:
            vehicles = [x for x in world.get_actors() if isinstance(x, carla.libcarla.Vehicle)]
            if len(vehicles) == 0:
                print('No vehicles found')
                sys.exit(1)
            self.car = vehicles[0]

        self.cur_weather = world.get_weather()

        self.map=world.get_map()

        self.lastPos = carla.Location(x=0, y=0, z=0)
        if EN_EGO:
            self.lastPos = self.car.get_location()
        
        
        self.screen=screen
        self.loaded_thumbnails=loaded_thumbnails
        
        
        self.roads=roads
        self.bg=bg
        self.myfont=myfont
        sprite = pygame.image.load('../common/map_sprite.png')
        self.sprite = pygame.transform.scale(sprite, (22, 44))
        self.weather_buttons=weather_buttons
        self.bgw, self.bgh = bg.get_rect().size

        self.isUserTyping = False
        self.userinput = ''

        self.placingBp = None
        self.firstClick = None


        self.draggingWeatherIdx = -1
        self.dragWeatherStart = None
        self.dragWeatherInitialVal = None

        self.makingPeds = False
        self.pedZoneFirstClick = None
        self.numPedsToMake = 0

        self.makingCars = False
        self.carZoneFirstClick = None
        self.numCarsToMake = 0
        
        self.prop_instances=[]
        
        self.dragging = -1
        
        self.outbox=outbox
        self.logfn=logfn
    def set_mapscale(self,mapscale,xint,yint):
        self.mapscale=mapscale
        self.xint=xint
        self.yint=yint

    def get_mouse_pos(self):
        mpos = pygame.mouse.get_pos()
        mpos=(mpos[0]-720,mpos[1])
        return mpos
        

    def render (self):
        '''
        render the mapview and the buttons of ad_wizard
        '''
    

        for ptype, pos, _ in self.prop_instances:
            tsurf = self.loaded_thumbnails[ptype]
            self.screen.blit(tsurf, (pos[0] - tsurf.get_width()//2, pos[1] - tsurf.get_height()//2))

        if EN_EGO:
            trans = self.car.get_transform()
            vloc = trans.location

            vpix = worldToPixel(vloc,self.mapscale,self.xint,self.yint)
            pygame.draw.line(self.bg, (255, 0, 0), worldToPixel(self.lastPos,self.mapscale,self.xint,self.yint), vpix)
            self.lastPos = vloc

            dists = [line_seg_to_point(*ls, (vloc.x, vloc.y)) for name, ls in self.roads]
            roadname = self.roads[argmin(dists)][0]
            head = headings[floor(fmod(trans.rotation.yaw + 22.5, 360)/45)]
            textsurf = self.myfont.render('Heading %s on %s' % (head, roadname), True, (255, 255, 255))

            sprite_rot = pygame.transform.rotate(self.sprite, 270 - trans.rotation.yaw)
            rotated_size = sprite_rot.get_rect().size
            self.screen.blit(sprite_rot, (vpix[0] - rotated_size[0] // 2, vpix[1] - rotated_size[1] // 2))

            self.screen.blit(textsurf, (10, 10))


        # update text box
        if self.draggingWeatherIdx != -1:
            mpos = self.get_mouse_pos()
            pname = WEATHER_PARAM_NAMES[self.draggingWeatherIdx]
            newval = self.dragWeatherInitialVal + int((mpos[0] - self.dragWeatherStart[0])*WEATHER_DRAG_FACTOR)
            self.weather_buttons[self.draggingWeatherIdx].set_text('%s (%.1f)' % (pname, newval))
        mpos = self.get_mouse_pos()

        t=carla.Location()
        t.x=pixelToWorld(mpos,self.mapscale,self.xint,self.yint)[0]
        t.y=pixelToWorld(mpos,self.mapscale,self.xint,self.yint)[1]
        t.z=0
        waypoint=self.map.get_waypoint(t)
        road_id=waypoint.road_id
        if waypoint.is_junction:
            road_id=waypoint.get_junction().id
        textsurf = self.myfont.render('(%ds)' % road_id, True, (255,255,255))
        self.screen.blit(textsurf, (mpos[0]+10, mpos[1]+10))

        if self.firstClick:
            mpos = self.get_mouse_pos()
            pygame.draw.line(self.screen, (0, 255, 0), self.firstClick, mpos)

            possurf = self.myfont.render('(%.1f, %.1f)'
                                    % pixelToWorld(self.firstClick, self.mapscale,self.xint,self.yint), True, (255,255,255))
            angsurf = self.myfont.render(u'%d\N{DEGREE SIGN}'
                                    % ((90 + int(np.arctan2(mpos[1]-self.firstClick[1],
                                                            mpos[0]-self.firstClick[0])*180/np.pi)) % 360),
                                    True, (255, 255, 255))
            self.screen.blit(possurf, (self.firstClick[0]+10, self.firstClick[1]+10))
            self.screen.blit(angsurf, (mpos[0], mpos[1]-20))

        elif self.placingBp:
            mpos = self.get_mouse_pos()

            textsurf = self.myfont.render('(%.1f, %.1f)' % pixelToWorld(mpos,self.mapscale,self.xint,self.yint), True, (255,255,255))
            self.screen.blit(textsurf, (mpos[0]+10, mpos[1]+10))

        if self.makingPeds:
            mpos = self.get_mouse_pos()
            textsurf = self.myfont.render('(%.1f, %.1f)' % pixelToWorld(mpos,self.mapscale,self.xint,self.yint), True, (255,255,255))
            self.screen.blit(textsurf, (mpos[0]+10, mpos[1]+10))
            if self.pedZoneFirstClick:
                pygame.draw.rect(self.screen, (0, 255, 0),
                                 pygame.Rect(min(self.pedZoneFirstClick[0], mpos[0]),
                                             min(self.pedZoneFirstClick[1], mpos[1]),
                                             abs(mpos[0] - self.pedZoneFirstClick[0]),
                                             abs(mpos[1] - self.pedZoneFirstClick[1])), 2)

        if self.makingCars:
            mpos = self.get_mouse_pos()
            textsurf = self.myfont.render('(%.1f, %.1f)' % pixelToWorld(mpos,self.mapscale,self.xint,self.yint), True, (255, 255, 255))
            self.screen.blit(textsurf, (mpos[0]+10, mpos[1]+10))
            if self.carZoneFirstClick:
                pygame.draw.rect(self.screen, (0, 255, 0),
                                 pygame.Rect(min(self.carZoneFirstClick[0], mpos[0]),
                                             min(self.carZoneFirstClick[1], mpos[1]),
                                             abs(mpos[0] - self.carZoneFirstClick[0]),
                                             abs(mpos[1] - self.carZoneFirstClick[1])), 2)

        for wb in self.weather_buttons:
            wb.render(self.screen)

        if self.isUserTyping:
            inputsurf = self.myfont.render('Input: ' + self.userinput, True, (255, 255, 255))
            self.screen.blit(inputsurf, (10, self.bgh - 20))

    def send_text(self,textin,chunks):

        '''
        react to input text
        '''
        if textin[0] == ':':
            prop = textin[1:]
            if prop not in props:
                print("Can't find blueprint for %s" % prop)
            else:
                self.placingBp = prop
                print('Click map to insert %s' % prop)

        elif textin[0] == '@':
            print('sending: %s' % textin[1:])
            self.outbox.put('dorothy:ad_wizard:message:' + textin[1:].strip())

        elif textin[0] == 't':
            print('sending trigger signal' )
            self.outbox.put('dorothy:ad_wizard:action:trigger')

        elif textin[0] == 'd':
            print('sending delete signal' )
            self.outbox.put('dorothy:ad_wizard:action:delete')

        elif textin[0] == 'c':
            print('sending change signal' )
            self.outbox.put('dorothy:ad_wizard:action:change')

        elif textin.startswith('!people'):
            print('Click first corner of people zone')
            self.makingPeds = True
            self.numPedsToMake = int(chunks[1]) if len(chunks) > 1 else 10

        elif textin.startswith('!cars'):
            print('Click first corner of car zone')
            self.makingCars = True
            self.numCarsToMake = int(chunks[1]) if len(chunks) > 1 else 10

        elif textin == '#bluron':
            self.outbox.put('co_wizard:ad_wizard:bluron')
            print('blur on')

        elif textin == '#bluroff':
            self.outbox.put('co_wizard:ad_wizard:bluroff')
            print('blur off')

        elif textin == '#nogpson':
            self.outbox.put('dorothy:ad_wizard:nogpson')
            self.outbox.put('co_wizard:ad_wizard:nogpson')
            print('no gps on')

        elif textin == '#nogpsoff':
            self.outbox.put('dorothy:ad_wizard:nogpsoff')
            self.outbox.put('co_wizard:ad_wizard:nogpsoff')
            print('no gps off')

        elif textin == '#noviewon':
            self.outbox.put('dorothy:ad_wizard:noviewon')
            print('no view on ')

        elif textin == '#noviewoff':
            self.outbox.put('dorothy:ad_wizard:noviewoff')
            print('no view off')

        else:
            print('Unrecognized directive: "%c"' % textin[0])



    def makepeds(self,mpos,frame):
        '''
        spawn pedestrians in a selected rectangle (only on sidewalks), there may be spawn failures
        '''
        if not self.pedZoneFirstClick:
            self.pedZoneFirstClick = mpos
            print('Click other corner of rect')

        else:
            pos = mpos
            xmin, xmax, ymin, ymax = min(pos[0], self.pedZoneFirstClick[0]), \
                                        max(pos[0], self.pedZoneFirstClick[0]), \
                                        min(pos[1], self.pedZoneFirstClick[1]), \
                                        max(pos[1], self.pedZoneFirstClick[1])
            xminw, yminw = pixelToWorld((xmin, ymin),self.mapscale,self.xint,self.yint)
            xmaxw, ymaxw = pixelToWorld((xmax, ymax),self.mapscale,self.xint,self.yint)
            #area = (xmaxw-xminw)*(ymaxw-yminw)
            #numpeds = int(area*0.1)

            print('Making %d peds' % self.numPedsToMake)
            pedctrl.createPeds(self.client, self.numPedsToMake, (xminw, xmaxw, yminw, ymaxw), sync=False)
            with open(self.logfn, 'a') as f:
                f.write('%.3f, %d, "make_peds", %d, %.3f, %.3f, %.3f, %.3f\n'
                        % (time.time(), frame, self.numPedsToMake, xminw, xmaxw, yminw, ymaxw))
            self.makingPeds = False
            self.pedZoneFirstClick = None

            

    def makecars(self,mpos,frame):
        '''
        spawn cars in a selected rectangle (only on roads), there may be spawn failures
        '''
        if not self.carZoneFirstClick:
            self.carZoneFirstClick = mpos
            print('Click other corner of rect')

        else:
            pos = mpos
            xmin, xmax, ymin, ymax = min(pos[0], self.carZoneFirstClick[0]), \
                                        max(pos[0], self.carZoneFirstClick[0]), \
                                        min(pos[1], self.carZoneFirstClick[1]), \
                                        max(pos[1], self.carZoneFirstClick[1])
            xminw, yminw = pixelToWorld((xmin, ymin),self.mapscale,self.xint,self.yint)
            xmaxw, ymaxw = pixelToWorld((xmax, ymax),self.mapscale,self.xint,self.yint)

            print('Making %d cars' % self.numCarsToMake)
            pedctrl.createVehicles(self.client, self.numCarsToMake,
                                    spawn_rect=(xminw, xmaxw, yminw, ymaxw), sync=False)
            with open(self.logfn, 'a') as f:
                f.write('%.3f, %d, "make_cars", %d, %.3f, %.3f, %.3f, %.3f\n'
                        % (time.time(), frame, self.numCarsToMake, xminw, xmaxw, yminw, ymaxw))
            self.makingCars = False
            self.carZoneFirstClick = None
    def makebp(self,mpos,frame):
        '''
        spawn selected blueprint. Mostly used to block the road
        '''
        
        if not self.firstClick:
            self.firstClick = mpos
        else:
            # second click (orientation)
            secondClick = mpos
            theta = np.arctan2(secondClick[1]-self.firstClick[1], secondClick[0]-self.firstClick[0])

            prop_name, origtf = props[self.placingBp]
            bp = self.world.get_blueprint_library().find(prop_name)

            mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
            mat = mat.dot(transform2M(radians(origtf.rotation.pitch),
                                        radians(origtf.rotation.yaw),
                                        radians(origtf.rotation.roll)))
            newpitch, newyaw, newroll = mToE(mat)

            tf = carla.Transform()
            tf.location.x = (self.firstClick[0] - self.xint)/self.mapscale
            tf.location.y = (self.firstClick[1] - self.yint)/self.mapscale
            tf.rotation.pitch = newpitch*180/np.pi
            tf.rotation.yaw = newyaw*180/np.pi
            tf.rotation.roll = newroll*180/np.pi

            closest_wp = self.map.get_waypoint(tf.location,
                                                        project_to_road=True,
                                                        lane_type=carla.LaneType.Any)
            # use Z as the offset
            tf.location.z = closest_wp.transform.location.z + origtf.location.z

            ac = self.world.try_spawn_actor(bp, tf)
            if ac:
                print('Actor spawn successful')
                print('Transform: %s, %s' % (str(tf.location), str(tf.rotation)))

                ac.set_simulate_physics(False)

                if self.placingBp not in self.loaded_thumbnails:
                    self.loaded_thumbnails[self.placingBp] = pygame.image.load(
                        os.path.join('../common/thumbnails', prop_name.replace('.', '_') + '.png'))
                self.prop_instances.append([self.placingBp, self.firstClick, ac])

                with open(self.logfn, 'a') as f:
                    f.write('%.3f, %d, "prop_create", "%s", %d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n'
                            % (time.time(), frame, prop_name, ac.id,
                                tf.location.x, tf.location.y, tf.location.z,
                                tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll))
            else:
                print('spawn failed')
            self.placingBp = None
            self.firstClick = None
            
            
    def on_event(self,frame):
        '''
        handle pygame events
        '''
        self.dragtf = None
        if self.dragging != -1:
            mpos = self.get_mouse_pos()
            wx, wy = pixelToWorld(mpos,self.mapscale,self.xint,self.yint)
            self.dragtf = self.prop_instances[self.dragging][2].get_transform()

            self.dragtf.location.x = wx
            self.dragtf.location.y = wy
            self.prop_instances[self.dragging][1] = mpos
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if self.isUserTyping:

                    # user done typing
                    if event.key == K_RETURN:
                        try:
                            textin = self.userinput.strip()
                            chunks = textin.split()
                            self.send_text(textin,chunks)
                        except ValueError:
                            pass

                        self.userinput = ''
                        self.isUserTyping = False

                    elif event.key == K_BACKSPACE:
                        self.userinput = self.userinput[:-1]

                    else:
                        self.userinput += event.unicode
                        # chr(event.key)

                elif event.key == K_SPACE:
                    self.isUserTyping = True

                elif event.key == K_ESCAPE:
                    if self.placingBp:
                        print('Cancel placement command')
                        self.placingBp = None
                        self.firstClick = None

            elif event.type == pygame.KEYUP:
                if self.dragging != -1:

                    if event.key == K_UP:
                        print('increment z')
                        self.dragtf.location.z += 1

                    elif event.key == K_DOWN:
                        print('decrement z')
                        self.dragtf.location.z -= 1

            elif event.type == pygame.MOUSEBUTTONUP:
                # left click
                mpos = self.get_mouse_pos()
                if event.button == 1:
                    if self.dragging != -1:
                        print('done dragging')

                        ac = self.prop_instances[self.dragging][2]
                        tf = ac.get_transform()

                        print('Transform: %s, %s' % (str(tf.location), str(tf.rotation)))
                        with open(self.logfn, 'a') as f:
                            f.write('%.3f, %d, "prop_move", %d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n'
                                    % (time.time(), frame, ac.id,
                                       tf.location.x, tf.location.y, tf.location.z,
                                       tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll))
                        self.dragging = -1
                        self.dragtf = None

                    if self.placingBp:
                        self.makebp(mpos,frame)
                    if self.makingPeds:
                        self.makepeds(mpos,frame)
                    if self.makingCars:
                        self.makecars(mpos,frame)
                    else:
                        if self.draggingWeatherIdx != -1 and abs(self.dragWeatherStart[0] - mpos[0]) > 10:
                            pname = WEATHER_PARAM_NAMES[self.draggingWeatherIdx]
                            newval = self.dragWeatherInitialVal + int((mpos[0] - self.dragWeatherStart[0])*WEATHER_DRAG_FACTOR)
                            # this factor controls how quickly clicking and dragging changes weather

                            print('setting %s to %.f' % (pname, newval))
                            setattr(self.cur_weather, pname, newval)
                            self.weather_buttons[self.draggingWeatherIdx].set_text('%s (%.1f)' % (pname, newval))
                            self.world.set_weather(self.cur_weather)

                            with open(self.logfn, 'a') as f:
                                f.write('%.3f, %d, "set_weather", "%s", %d\n'
                                        % (time.time(), frame, pname, int(newval)))
                        else:
                            for i, wb in enumerate(self.weather_buttons):
                                if wb.click(mpos):
                                    pname = WEATHER_PARAM_NAMES[i]
                                    newval = getattr(self.cur_weather, pname) - WEATHER_CLICK_INC
                                    print('setting %s to %.1f' % (pname, newval))
                                    setattr(self.cur_weather, pname, newval)
                                    wb.set_text(pname + ' (%.1f)' % newval)
                                    self.world.set_weather(self.cur_weather)
                                    with open(self.logfn, 'a') as f:
                                        f.write('%.3f, %d, "set_weather", "%s", %d\n'
                                                % (time.time(), frame, pname, int(newval)))

                        self.draggingWeatherIdx = -1

                # right click
                elif event.button == 3:
                    mpos = self.get_mouse_pos()
                    for i in range(len(self.prop_instances)):
                        ptype, pos, ac = self.prop_instances[i]
                        twidth, theight = self.loaded_thumbnails[ptype].get_size()
                        if abs(mpos[0] - pos[0]) < twidth//2 and abs(mpos[1] - pos[1]) < theight//2:
                            with open(self.logfn, 'a') as f:
                                f.write('%.3f, %d, "prop_destroy", %d\n'
                                        % (time.time(), frame, ac.id))
                            ac.destroy()
                            del self.prop_instances[i]
                            break

                    for i, wb in enumerate(self.weather_buttons):
                        if wb.click(mpos):
                            pname = WEATHER_PARAM_NAMES[i]
                            newval = getattr(self.cur_weather, pname) + WEATHER_CLICK_INC
                            print('setting %s to %.1f' % (pname, newval))
                            setattr(self.cur_weather, pname, newval)
                            wb.set_text(pname + ' (%.1f)' % newval)
                            self.world.set_weather(self.cur_weather)
                            with open(self.logfn, 'a') as f:
                                f.write('%.3f, %d, "set_weather", "%s", %d\n'
                                        % (time.time(), frame, pname, int(newval)))

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mpos = self.get_mouse_pos()
                    # left click
                    if not self.placingBp:
                        for i in range(len(self.prop_instances)):
                            ptype, pos, ac = self.prop_instances[i]
                            twidth, theight = self.loaded_thumbnails[ptype].get_size()
                            if abs(mpos[0] - pos[0]) < twidth//2 and abs(mpos[1] - pos[1]) < theight//2:
                                self.dragging = i
                                self.dragtf = None
                                break

                    for i, wb in enumerate(self.weather_buttons):
                        if wb.click(mpos):
                            self.draggingWeatherIdx = i
                            self.dragWeatherStart = mpos
                            self.dragWeatherInitialVal = getattr(self.cur_weather, WEATHER_PARAM_NAMES[i])
                            print('start dragging %s' % WEATHER_PARAM_NAMES[i])
                            break

        if self.dragging != -1 and self.dragtf:
            self.prop_instances[self.dragging][2].set_transform(self.dragtf)

def main(args):
    """
    Main method
    """

    def handleImage(image, surfq):
        # image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        surfq.put(pygame.surfarray.make_surface(array.swapaxes(0, 1)))
        
    DEBUG = args.debug
    loaded_thumbnails = {}
    fps = 30
    if not os.path.exists("ad_wizardlogs"):
        os.mkdir("ad_wizardlogs")
    # OBSOLETE: all happens in co_wizard
    logfn = 'ad_wizardlogs/ad_wizardlog_%d.csv' % time.time()
    with open(logfn, 'a') as f:
        f.write('systime, frame, action, args\n')

    client = carla.Client(args.carlahost, args.carlaport)
    client.set_timeout(10.0)
    world = client.get_world()
    
    cur_weather = world.get_weather()

    inbox, outbox = Queue(), Queue()
    msg_thread = threading.Thread(target=messageClient,
                                  args=(args.co_wizardhost, args.co_wizardport, inbox, outbox), daemon=True)
    msg_thread.start()
    with open(logfn, 'a') as f:
        f.write('%.3f, %d, "start"\n' % (time.time(), world.get_snapshot().frame))

    # Request config file from co_wizard
    outbox.put('co_wizard:ad_wizard:cam_rq') #request config file from co_wizard
    while 1:
        if not inbox.empty():
            msg = inbox.get()
            if msg.startswith('ad_wizard:co_wizard:sb:'):
                cam_ids = json.loads(msg[len('ad_wizard:co_wizard:sb:'):])
                print('cam_id retrieved')
                break
            else:
                print('ignoring message while waiting for cam_id: %s' % msg)
        print('Waiting to receive cam_id from co_wizard...')
        time.sleep(1)
        
    cam = world.get_actor(cam_ids["cam_top_id"])
    surfq = Queue(maxsize=1)
    cam.listen(lambda x: handleImage(x, surfq))
    outbox.put('co_wizard:ad_wizard:config_rq')
    while True:
        if not inbox.empty():
            msg = inbox.get()
            if msg.startswith('ad_wizard:co_wizard:config:'):
                simconfig = json.loads(msg[len('co_wizard:ad_wizard:config:'):])
                print('Config retrieved')
                break
            else:
                print('ignoring message while waiting for config: %s' % msg)
        print('Waiting to receive config from co_wizard...')
        time.sleep(1)

    mapscale = simconfig['map_metadata']['pixels_per_meter']
    xint = simconfig['map_metadata']['origin_pixel_coord_x']
    yint = simconfig['map_metadata']['origin_pixel_coord_y']
    print(mapscale)

    pygame.init()
    pygame.font.init()
    myfont = pygame.font.SysFont('Roboto Mono', 16)
    bigfont = pygame.font.SysFont('Roboto Mono', 30)
    clock = pygame.time.Clock()
    bg = pygame.image.load('../common/map_images/%s.jpg' % simconfig['map'])

    bgw, bgh = bg.get_rect().size
    scrw, scrh = bgw, bgh + 250
    roads = render_map_from_simconfig(bg, simconfig, bigfont)
    WIDTH, HEIGHT = 720,  720

    screen = pygame.display.set_mode((
            WIDTH + scrw, max(HEIGHT, scrh)),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
    map_screen=pygame.Surface((scrw,scrh))
    weather_buttons = []
    for i, p in enumerate(WEATHER_PARAM_NAMES):
        weather_buttons.append(Button(p + ' (%.1f)' % getattr(cur_weather, p),
                                        myfont, (scrw//3)*(i % 3) + (scrw//6),
                                        (250//4)*(i//3) + bgh + (250//8),
                                        scrw//3 - 20, 250//4 - 20))


    ad_wizard=Ad_wizard(client,world,map_screen,myfont,loaded_thumbnails,roads,bg,outbox,logfn,weather_buttons) 
    ad_wizard.set_mapscale(mapscale,xint,yint)
    while True:

        rerender = False
        imgsurf = None
        if not surfq.empty():
            surfin = surfq.get()
            if surfin:
                rerender = True
                imgsurf = surfin

        clock.tick(fps)
        frame = world.get_snapshot().frame
        # print(frame)

        map_screen.fill((0, 0, 0))
        map_screen.blit(bg, (0, 0))
        ad_wizard.render()
        ad_wizard.on_event(frame)
        if rerender:
            if imgsurf:
                screen.blit(imgsurf, (0, 0))
            screen.blit(map_screen,(WIDTH,0))

        pygame.display.update()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Witch Executable')
    argparser.add_argument(
        '-ch', '--carla-host',
        dest='carlahost',
        type=str,
        help='Hostname/IP for CARLA server',
        default='localhost')
    argparser.add_argument(
        '-cp', '--carla-port',
        dest='carlaport',
        type=int,
        help='Port for CARLA server',
        default=2000)
    argparser.add_argument(
        '-wh', '--co_wizard-host',
        dest='co_wizardhost',
        type=str,
        help='Hostname/IP for co_wizard',
        default='localhost')
    argparser.add_argument(
        '-wp', '--co_wizard-port',
        dest='co_wizardport',
        type=int,
        help='Port for co_wizard',
        default=6790)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')

    args = argparser.parse_args()
    main(args)
