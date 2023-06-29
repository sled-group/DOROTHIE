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
import sys
import argparse
import time
from queue import Queue

from math import sqrt, fmod, floor
from numpy import argmin

sys.path.insert(0, '../util')
from avutil import render_map_from_simconfig, line_seg_to_point


def runDorothyMap(carlahost, carlaport, map_cmd_queue, simconfig, standalone=False, frames_queue=None):

    headings = ['east', 'southeast', 'south', 'southwest', 'west', 'northwest', 'north', 'northeast']
    mapscale = simconfig['map_metadata']['pixels_per_meter']
    xint = simconfig['map_metadata']['origin_pixel_coord_x']
    yint = simconfig['map_metadata']['origin_pixel_coord_y']

    def worldToPixel(loc):
        if isinstance(loc, tuple) or isinstance(loc, list):
            return (int(loc[0] * mapscale + xint), int(loc[1] * mapscale + yint))
        return (int(loc.x * mapscale + xint), int(loc.y * mapscale + yint))

    client = carla.Client(carlahost, carlaport)  # create our own client b/c not thread safe
    client.set_timeout(10.0)
    world = client.get_world()

    while 1:
        vehicles = [x for x in world.get_actors() if
                    isinstance(x, carla.libcarla.Vehicle) and ('role_name', 'hero') in x.attributes.items()]
        if len(vehicles) > 0: break
        print('Waiting for vehicle...')
        time.sleep(2)
    car = vehicles[0]

    if standalone:
        pygame.init()
        pygame.font.init()
    clock = pygame.time.Clock()
    myfont = pygame.font.SysFont('Roboto Mono', 22)
    bigfont = pygame.font.SysFont('Roboto Mono', 30)

    bg = pygame.image.load('../common/map_images/%s.jpg' % simconfig['map'])
    sprite = pygame.image.load('../common/map_sprite.png')
    sprite = pygame.transform.scale(sprite, (22, 44))

    roads = render_map_from_simconfig(bg, simconfig, bigfont)

    if standalone:
        screen = pygame.display.set_mode(bg.get_rect().size)
    else:
        screen = pygame.Surface(bg.get_rect().size)
        assert frames_queue


    nogps = False
    fps = 30

    while True:
        clock.tick(fps)

        while not map_cmd_queue.empty():
            msg = map_cmd_queue.get()
            if msg == 'nogpson':
                nogps = True
            elif msg == 'nogpsoff':
                nogps = False

        trans = car.get_transform()
        vloc = trans.location

        vpix = worldToPixel(vloc)

        screen.blit(bg, (0, 0))
        if nogps:
            textsurf = myfont.render('ERROR: GPS signal lost', True, (255, 255, 255))
            screen.blit(textsurf, (10, 10))
        else:
            sprite_rot = pygame.transform.rotate(sprite, 270 - trans.rotation.yaw)
            rotated_size = sprite_rot.get_rect().size
            screen.blit(sprite_rot, (vpix[0] - rotated_size[0] // 2, vpix[1] - rotated_size[1] // 2))

            dists = [line_seg_to_point(*ls, (vloc.x, vloc.y)) for name, ls in roads]
            roadname = roads[argmin(dists)][0]
            head = headings[floor(fmod(trans.rotation.yaw + 22.5, 360) / 45)]
            textsurf = myfont.render('Heading %s on %s' % (head, roadname), True, (255, 255, 255))
            screen.blit(textsurf, (10, 10))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        if standalone:
            pygame.display.update()
        else:
            frames_queue.put(screen.copy())
