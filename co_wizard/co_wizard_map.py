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

"""
This is the map interface of the co_wizard operator.
"""

# ==============================================================================
# -- IMPORTS -------------------------------------------------------------------
# ==============================================================================

import carla
import pygame
from pygame.locals import K_z
import time
import os
from math import sqrt, fmod, floor
from numpy import argmin, arctan2, pi, inf
import sys
import argparse

sys.path.insert(0, '../util')
from avutil import render_map_from_simconfig, line_seg_to_point

DEBUG = False


def run_co_wizard_map(args,
                   simconfig, storyboard,
                   map_output_queue, map_input_queue, map_frames_queue, pygame_event_queue,
                   logfn, mouse_x_offset=0):

    headings = ['east', 'southeast', 'south', 'southwest', 'west', 'northwest', 'north', 'northeast']
    mapscale = simconfig['map_metadata']['pixels_per_meter']
    xint = simconfig['map_metadata']['origin_pixel_coord_x']
    yint = simconfig['map_metadata']['origin_pixel_coord_y']

    opt_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]

    def world_to_pixel(loc):
        if isinstance(loc, tuple) or isinstance(loc, list):
            return int(loc[0] * mapscale + xint), int(loc[1] * mapscale + yint)
        return int(loc.x * mapscale + xint), int(loc.y * mapscale + yint)

    def pixel_to_world(loc):
        if isinstance(loc, tuple) or isinstance(loc, list):
            return (int(loc[0] - xint) / mapscale), int((loc[1] - yint) / mapscale)
        return int((loc.x - xint) / mapscale), int((loc.y - yint) / mapscale)

    newPaths = None
    wps = []

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    while 1:
        vehicles = [x for x in world.get_actors() if
                    isinstance(x, carla.libcarla.Vehicle) and ('role_name', 'hero') in x.attributes.items()]
        if len(vehicles) > 0:
            break
        print('Waiting for vehicle...')
        time.sleep(2)
    car = vehicles[0]
    print(str(car))

    myfont = pygame.font.SysFont('Roboto Mono', 22)
    bigfont = pygame.font.SysFont('Roboto Mono', 30)
    bigbigfont = pygame.font.SysFont('Roboto Mono', 48)
    clock = pygame.time.Clock()

    bg = pygame.image.load('../common/map_images/%s.jpg' % simconfig['map'])
  
    # Scale the image to your needed size
    roads = render_map_from_simconfig(bg, simconfig, bigfont, hide_assets=(
        set(storyboard['hidden_from_co_wizard']) if 'hidden_from_co_wizard' in storyboard else set()))

    sprite = pygame.image.load('../common/map_sprite.png')
    sprite = pygame.transform.scale(sprite, (22, 44))

    screen = pygame.Surface(bg.get_rect().size)

    # get list of all junctions
    carlamap = world.get_map()
    allwps = carlamap.generate_waypoints(5)
    jctm = {w.get_junction().id: w.get_junction() for w in allwps if w.is_junction}

    jcts = list(jctm.values())
    bbsm = {}
    for jct in jcts:
        grp = None
        for g in simconfig['map_metadata']['junction_groups']:
            if jct.id in g:
                grp = tuple(g)
                break

        if not grp: grp = jct.id

        wbb = (jct.bounding_box.location.x - jct.bounding_box.extent.x,
               jct.bounding_box.location.x + jct.bounding_box.extent.x,
               jct.bounding_box.location.y - jct.bounding_box.extent.y,
               jct.bounding_box.location.y + jct.bounding_box.extent.y)
        l, t = world_to_pixel((wbb[0], wbb[2]))
        r, b = world_to_pixel((wbb[1], wbb[3]))

        if grp not in bbsm:
            bbsm[grp] = (l, r, t, b)

        else:
            # update existing entry based on bounds
            pl, pr, pt, pb = bbsm[grp]
            bbsm[grp] = (min(l, pl), max(r, pr), min(t, pt), max(b, pb))

    assetbbs = []
    for ast in simconfig['assets']:
        px, py = world_to_pixel(ast['location'][:2])
        assetbbs.append(('%s_%s_%s_%s' % (ast['type'], *ast['location']), (px, py)))

    planmode = False
    curplan = []
    plangoal = None

    lastPos = car.get_location()

    trajSurface = None
    optPositions = {}

    # if ad_wizard causes GPS loss
    nogps = False
    fps = 30

    while True:
        clock.tick(fps)
        ss = world.get_snapshot()

        while not map_input_queue.empty():
            cmd, pars = map_input_queue.get()

            # this tells us what the path options are at the next intersection
            if cmd == 'path_options':
                newPaths = pars

            # this shows the currently selected waypoint path up to the next intersection
            elif cmd == 'waypoint_path':
                # pars is wps, a list of (x, y) wp coords (use every other for nice rendering)
                with open(logfn, 'a') as f:
                    f.write('%.3f, %d, "planner", "agent", "plan", "plan_turn", "%s"\n'
                            % (time.time(), ss.frame, str(pars)))

            elif cmd == 'new_road_id':
                with open(logfn, 'a') as f:
                    f.write('%.3f, %d, "planner", "agent", "status", "at_junction", %d\n'
                            % (time.time(), ss.frame, pars))

            elif cmd == 'toggle_plan_mode':
                if not planmode:  # enter plan mode
                    print('Enter plan mode, clear plan')
                    planmode = True
                    curplan.clear()
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "agent", "plan", "start_plan", "NONE"\n'
                                % (time.time(), ss.frame))

                # exit plan mode
                else:
                    print('Exit plan mode')
                    planmode = False
                    with open(logfn, 'a') as f:
                        f.write('%.3f, %d, "co_wizard", "agent", "plan", "new_plan", "%s"\n'
                                % (time.time(), ss.frame, str(curplan)))

            if cmd == 'nogpson':
                nogps = True

            elif cmd == 'nogpsoff':
                nogps = False

        trans = car.get_transform()
        vloc = trans.location

        vpix = world_to_pixel(vloc)
        pygame.draw.line(bg, (255, 0, 0), world_to_pixel(lastPos), vpix)
        lastPos = vloc

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

        for wp in wps[::2]:  # draw every other
            pygame.draw.circle(screen, (255, 255, 255), world_to_pixel(wp), 3)

        if newPaths:
            trajSurface = pygame.Surface(bg.get_size())
            trajSurface.set_colorkey((0, 0, 0))
            optPositions = {}
            for trajIdx, traj in enumerate(newPaths):
                curColor = opt_colors[trajIdx]
                lastWp = None
                lastRoadID = None
                for wpIdx, wpCoords in enumerate(traj):
                    curWp = world_to_pixel(wpCoords)
                    if lastWp:
                        pygame.draw.line(trajSurface, curColor, curWp, lastWp)
                    lastWp = curWp
                    lastRoadID = wpCoords[2]
                pygame.draw.circle(trajSurface, curColor, lastWp, 10)
                optPositions[lastRoadID] = lastWp
            print(optPositions)
            newPaths = None

        if trajSurface:
            screen.blit(trajSurface, (0, 0))

        if planmode:
            for jid, (l, r, t, b) in bbsm.items():
                pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(l, t, r - l, b - t), 2)


        if curplan:
            for i, pj in enumerate(curplan):
                l, r, t, b = bbsm[pj]
                txtsurf = bigbigfont.render(str(i + 1), True, (255, 0, 0))
                screen.blit(txtsurf,
                            ((l + r) // 2 - txtsurf.get_width() // 2, (t + b) // 2 - txtsurf.get_height() // 2))

        if plangoal:
            pygame.draw.circle(screen, (0, 255, 0), plangoal[1], 10, 2)

        map_frames_queue.put(screen.copy())

        # we can't have both the co_wizard map and co_wizard 3d consuming pygame events
        # because they are cleared after each access,
        # so the main co_wizard copies them to a queue for the map
        while not pygame_event_queue.empty():
            event = pygame_event_queue.get()

            if event.type == pygame.MOUSEBUTTONUP:
                mpos = pygame.mouse.get_pos()
                mx, my = mpos[0] + mouse_x_offset, mpos[1]

                for idx, (road_id, opos) in enumerate(optPositions.items()):
                    if (mx - opos[0]) ** 2 + (my - opos[1]) ** 2 < 100:
                        print('selected path %d: road id = %d' % (idx, road_id))

                        map_output_queue.put(('chosen_path', idx))
                        with open(logfn, 'a') as f:
                            f.write('%.3f, %d, "co_wizard", "agent", "plan", "next_road", %d\n'
                                    % (time.time(), ss.frame, road_id))
                        break

                if planmode:

                    # Left Click
                    if event.button == 1:
                        for jid, (l, r, t, b) in bbsm.items():
                            if mx >= l and mx <= r and my >= t and my <= b:
                                curplan.append(jid)
                                print('Adding jct with id %s to plan' % str(jid))
                                print('Current whole plan: %s' % str(curplan))
                                break

                    # Right Click
                    elif event.button == 3:
                        found_asset_goal = False
                        for astname, (px, py) in assetbbs:
                            if abs(mx - px) < 50 and abs(my - py) < 50:
                                found_asset_goal = True
                                plangoal = (astname, (px, py))
                                with open(logfn, 'a') as f:
                                    f.write('%.3f, %d, "co_wizard", "agent", "plan", "new_plan_goal_asset", "%s"\n'
                                            % (time.time(), ss.frame, astname))
                                break

                        if not found_asset_goal:
                            plangoal = ('', (mx, my))
                            with open(logfn, 'a') as f:
                                f.write('%.3f, %d, "co_wizard", "agent", "plan", "new_plan_goal_location", "%s"\n'
                                        % (time.time(), ss.frame, str(pixel_to_world((mx, my)))))

            elif event.type == pygame.KEYUP:
                if event.key == K_z:
                    if planmode:
                        if curplan:
                            curplan.pop(-1)
                        else:
                            print('Note: no plan elements to pop')
