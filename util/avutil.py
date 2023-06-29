#!/usr/bin/python3

# ==============================================================================
# -- COPYRIGHT -----------------------------------------------------------------
# ==============================================================================

# Copyright (c) 2022 SLED Lab, EECS, University of Michigan.
# authors: Ben VanDerPloeg (bensvdp@umich.edu),
#          Martin Ziqiao Ma (marstin@umich.edu),
#          Owen Yidong Huang (owenhji@umich.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import pygame
import os
from math import sqrt
import json

DEFAULT_IMAGE_SIZE = (30,30)

def line_seg_to_point(A, B, E):
    """
    Credit:
    https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
    """
    # vector AB
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]

    # vector BP
    BE = [None, None]
    BE[0] = E[0] - B[0]
    BE[1] = E[1] - B[1]

    # vector AP
    AE = [None, None]
    AE[0] = E[0] - A[0]
    AE[1] = E[1] - A[1]

    # Variables to store dot product
    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]

    # Minimum distance from
    # point E to the line segment
    reqAns = 0

    # Case 1
    if AB_BE > 0:

        # Finding the magnitude
        y = E[1] - B[1]
        x = E[0] - B[0]
        reqAns = sqrt(x * x + y * y)

    # Case 2
    elif AB_AE < 0:
        y = E[1] - A[1]
        x = E[0] - A[0]
        reqAns = sqrt(x * x + y * y)

    # Case 3
    else:
        # Finding the perpendicular distance
        x1 = AB[0]
        y1 = AB[1]
        x2 = AE[0]
        y2 = AE[1]
        mod = sqrt(x1 * x1 + y1 * y1)
        reqAns = abs(x1 * y2 - y1 * x2) / mod

    return reqAns


#modifies map by adding roads, returns list of roads and positions
def render_map_from_simconfig(map, simconfig, font, loaded_thumbnails={}, hide_assets=set()):
    mapscale = simconfig['map_metadata']['pixels_per_meter']
    xint = simconfig['map_metadata']['origin_pixel_coord_x']
    yint = simconfig['map_metadata']['origin_pixel_coord_y']

    def worldToPixel(loc):
        return (int(loc[0]*mapscale + xint), int(loc[1]*mapscale + yint))

    roads = []
    for street_idx, street_seg in enumerate(simconfig['map_metadata']['road_coords']):
        name = simconfig['street_names'][street_idx] + ' ' + simconfig['map_metadata']['road_designations'][street_idx]
        roads.append((name, street_seg))

        sx, sy = worldToPixel(street_seg[0])
        ex, ey = worldToPixel(street_seg[1])

        txtsurf = font.render(name, True, (0, 255, 0))
        wmax = int(0.8*sqrt((sx - ex)**2 + (sy - ey)**2))
        if txtsurf.get_width() > wmax:
            txtsurf = pygame.transform.scale(txtsurf, (wmax, wmax*txtsurf.get_height()//txtsurf.get_width()))

        px, py, ptheta = simconfig['map_metadata']['road_label_locations'][street_idx]
        txtsurf = pygame.transform.rotate(txtsurf, ptheta)
        map.blit(txtsurf, (px - txtsurf.get_width()//2, py - txtsurf.get_height()//2))

        # txtangle = arctan2(ey - sy, ex - sx)
        # if abs(txtangle) > pi/2: txtangle += pi #never more than 90 degree rotation in either direction - text should be upright
        # txtsurf = myfont.render(name, True, (0, 255, 0))
        # txtsurf = pygame.transform.rotate(txtsurf, txtangle*180/pi)
        #
        # bg.blit(txtsurf, ((sx + ex)//2 - txtsurf.get_width()//2, (sy + ey)//2 - txtsurf.get_height()//2))

    if hide_assets:
        with open('../common/asset_metadata.json') as f:
            asset_md = json.load(f)

    for asset_instance in simconfig['assets']:
        bp = asset_instance['type']

        if hide_assets:
            aname = asset_md[bp]['name']
            if aname in hide_assets:
                print('NOTE: hiding asset %s (%s) from co_wizard' % (aname, bp))
                continue

        if bp not in loaded_thumbnails:
            image=pygame.image.load(os.path.join('../common/thumbnails', bp.replace('.', '_') + '.png'))
            image=image = pygame.transform.scale(image, DEFAULT_IMAGE_SIZE)
            loaded_thumbnails[bp] = image
        thumbnail = loaded_thumbnails[bp]

        pixelpos = worldToPixel(asset_instance['location'][:2])
        map.blit(thumbnail, (pixelpos[0] - thumbnail.get_width()//2, pixelpos[1] - thumbnail.get_height()//2))

    return roads


