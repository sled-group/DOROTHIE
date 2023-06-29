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
from pygame.locals import K_t
import numpy as np
import time
import threading
from queue import Queue
import socket
import argparse
import dorothy_speech_client
from scipy.ndimage import gaussian_filter
import os
import json
import re

import dorothy_map


if not os.path.exists('dorothylogs'):
    os.mkdir('dorothylogs')

episode_timestamp = time.time()
os.mkdir('dorothylogs/log_%d' % episode_timestamp)
logfn = 'dorothylogs/log_%d/dorothylogs_%d.csv' % (episode_timestamp, episode_timestamp)
with open(logfn, 'a') as f:
    f.write('systime, frame, utterance\n')

audio_file = 'dorothylogs/log_%d/dorothy_audio' % episode_timestamp
if not os.path.exists(audio_file):
    os.mkdir(audio_file)


def co_wizardTCP(dorothy_inbox, dorothy_outbox, speech_inbox, speech_outbox, host, port):
    while True:
        textIn = ''
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            print('Attempting to connect to host "%s" on port %d' % (host, port))
            sock.connect((host, port))
            sock.setblocking(False)

            while True:
                try:
                    # throws timeout exception if none available
                    textIn += str(sock.recv(1024), "utf-8")
                    if '\n' in textIn:
                        endpos = textIn.index('\n')
                        msg = textIn[:endpos]
                        textIn = textIn[endpos+1:]

                        if msg.strip() == 'keepalive':
                            continue
                        print('received: %s' % msg)

                        if msg.startswith('dorothy:co_wizard:message:'):
                            speech_inbox.put(msg[len('dorothy:co_wizard:message:'):])
                        else:
                            dorothy_inbox.put(msg)
                except:
                    pass

                if not speech_outbox.empty():
                    sock.sendall(
                        bytes('co_wizard:dorothy:message:' + speech_outbox.get().strip() + '\n', 'utf-8'))

                if not dorothy_outbox.empty():
                    tosend = 'co_wizard:dorothy:' + dorothy_outbox.get().strip() + '\n'
                    print('Dorothy sending: %s' % tosend)
                    sock.sendall(
                        bytes(tosend, 'utf-8'))

                time.sleep(0.01)


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    """
    Main method
    """
    def handleImage(image, surfq):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        surfq.put(pygame.surfarray.make_surface(array.swapaxes(0, 1)))

    argparser = argparse.ArgumentParser(description='Dorothy Ego View Executable')
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
        default=6791)
    argparser.add_argument(
        '--no-map',
        action='store_false',
        dest='usemap',
        help='Disable automatic launch of Dorothy map')

    args = argparser.parse_args()

    cl = carla.Client(args.carlahost, args.carlaport)
    world = cl.get_world()


    pygame.init()
    pygame.font.init()
    myfont = pygame.font.SysFont('Roboto Mono', 48)


    done = False

    ad_wizard_msg = None
    ad_wizard_msg_expire = 0

    speech_ctrl_q, speech_inbox, speech_outbox = Queue(), Queue(), Queue()
    speech_thread = threading.Thread(target=dorothy_speech_client.run,
                                     args=(speech_ctrl_q, speech_inbox, speech_outbox, episode_timestamp), daemon=True)
    speech_thread.start()

    dorothy_inbox, dorothy_outbox = Queue(), Queue()
    wmthread = threading.Thread(target=co_wizardTCP,
                                args=(dorothy_inbox, dorothy_outbox,
                                      speech_inbox, speech_outbox,
                                      args.co_wizardhost, args.co_wizardport), daemon=True)
    wmthread.start()

    # request config file from co_wizard
    dorothy_outbox.put('co_wizard:dorothy:cam_rq') #request config file from co_wizard
    while 1:
        if not dorothy_inbox.empty():
            msg = dorothy_inbox.get()
            if msg.startswith('dorothy:co_wizard:sb:'):
                cam_ids = json.loads(msg[len('dorothy:co_wizard:sb:'):])
                print('cam_id retrieved')
                break
            else:
                print('ignoring message while waiting for cam_id: %s' % msg)
        print('Waiting to receive cam_id from co_wizard...')
        time.sleep(1)
        
    cam = world.get_actor(cam_ids["cam_id"])
    surfq = Queue(maxsize=1)
    cam.listen(lambda x: handleImage(x, surfq))
    dorothy_outbox.put('co_wizard:dorothy:config_rq')
    while 1:
        if not dorothy_inbox.empty():
            msg = dorothy_inbox.get()
            if msg.startswith('dorothy:co_wizard:config:'):
                simconfig = json.loads(msg[len('dorothy:co_wizard:config:'):])
                print('Config retrieved')
                break
            else:
                print('ignoring message while waiting for config: %s' % msg)
        print('Waiting to receive config from co_wizard...')
        time.sleep(1)
    dorothy_outbox.put('co_wizard:dorothy:sb_rq') #request config file from co_wizard
    while 1:
        if not dorothy_inbox.empty():
            msg = dorothy_inbox.get()
            if msg.startswith('dorothy:co_wizard:sb:'):
                storyboard = json.loads(msg[len('dorothy:co_wizard:sb:'):])
                print('Storyboard retrieved')
                break
            else:
                print('ignoring message while waiting for storyboard: %s' % msg)
        print('Waiting to receive storyboard from co_wizard...')
        time.sleep(1)

    map_cmd_queue, map_frames_queue = Queue(), Queue(maxsize=1)
    if args.usemap:
        map_thread = threading.Thread(target=dorothy_map.runDorothyMap,
                                      args=(args.carlahost, args.carlaport, map_cmd_queue, simconfig),
                                      kwargs={'frames_queue': map_frames_queue}, daemon=True)
        map_thread.start()
        time.sleep(2)

    MAPWIDTH, MAPHEIGHT = 679, 670
    WIDTH, HEIGHT = 1280,  720

    if args.usemap:
        screen = pygame.display.set_mode((
            WIDTH + MAPWIDTH, max(HEIGHT, MAPHEIGHT)),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
    else:
        screen = pygame.display.set_mode(
            (WIDTH, HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Dorothy")

    text = [goal["description"] for goal in storyboard["subgoals"]]
    accomplished = [False for t in text]
    ad_wizard_blur = False
    ad_wizard_noview = False
    parked = True
    accomplishednum = 0
    triggered = False
    deleted = False

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                done = True
                break
            elif e.type == pygame.KEYDOWN:
                if e.key == K_t:
                    speech_ctrl_q.put('start')
                    with open(logfn, 'a') as f:
                        f.write('%d, %d, start\n'
                                % (time.time(), world.get_snapshot().frame))

            elif e.type == pygame.KEYUP:
                if e.key == K_t:
                    speech_ctrl_q.put('stop')
                    with open(logfn, 'a') as f:
                        f.write('%d, %d, end\n'
                                % (time.time(), world.get_snapshot().frame))
        if done:
            break

        rerender = False
        imgsurf = None

        if not surfq.empty():
            surfin = surfq.get()
            if surfin:
                rerender = True
                imgsurf = surfin

        if not dorothy_inbox.empty():
            msg = dorothy_inbox.get().strip()

            if msg.startswith('dorothy:ad_wizard:message:'):
                ad_wizard_msg = msg[len('dorothy:ad_wizard:message:'):]
                # ad_wizard messages show for 10 seconds
                ad_wizard_msg_expire = time.time() + 10
                print('received ad_wizard message: "%s"' % ad_wizard_msg)

            elif msg == 'dorothy:ad_wizard:bluron':
                ad_wizard_blur = True

            elif msg == 'dorothy:ad_wizard:bluroff':
                ad_wizard_blur = False

            elif msg == 'dorothy:ad_wizard:noviewon':
                ad_wizard_noview = True
                ad_wizard_blur = False

            elif msg == 'dorothy:ad_wizard:noviewoff':
                ad_wizard_noview = False

            elif msg == 'dorothy:ad_wizard:action:trigger':
                triggered = True

            elif msg == 'dorothy:ad_wizard:action:delete':
                deleted = True
                print("deleted")
            elif msg == 'dorothy:ad_wizard:action:change':
                for i in range(len(storyboard["subgoals"])):
                    if storyboard["subgoals"][i]['change']!="":
                        storyboard["subgoals"][i]["description"]=storyboard["subgoals"][i]['change']

            elif msg == 'dorothy:ad_wizard:nogpson':
                if args.usemap:
                    map_cmd_queue.put('nogpson')

            elif msg == 'dorothy:ad_wizard:nogpsoff':
                if args.usemap:
                    map_cmd_queue.put('nogpsoff')

            elif msg == 'dorothy:co_wizard:parkon':
                parked = True

            elif msg == 'dorothy:co_wizard:parkoff':
                parked = False

            elif msg.startswith('dorothy:co_wizard:sb_reach:'):
                arrived_num = int(msg[len('dorothy:co_wizard:sb_reach:'):])
                accomplished[arrived_num] = True

        if rerender:
            if imgsurf:
                if ad_wizard_noview:
                    screen.fill((0,0,0))
                    textsurf = myfont.render('Sensor Malfunction', True, (255, 0, 0))
                    screen.blit(textsurf, (WIDTH // 2 - 50, HEIGHT // 3))
                else:
                    screen.blit(imgsurf, (0, 0))

            if ad_wizard_blur:
                img_np = pygame.surfarray.array3d(screen)
                img_np_blurred = gaussian_filter(img_np, (9, 9, 1))
                pygame.surfarray.blit_array(screen, img_np_blurred)

            if time.time() < ad_wizard_msg_expire:
                txtsurf = myfont.render(ad_wizard_msg, True, (255,255,255))
                screen.blit(txtsurf, (30, 30))

            if parked:
                textsurf = myfont.render('Parked', True, (255, 0, 0))
                screen.blit(textsurf, (WIDTH // 2, HEIGHT // 2))

            if args.usemap and not map_frames_queue.empty():
                # right align with 3D view
                map_surf = map_frames_queue.get()
                screen.blit(map_surf, (WIDTH, 0))

            if args.usemap:
                storyfont = pygame.font.Font(None,26)
                text = [(storyboard["subgoals"][i]["description"],i)
                        for i in range(len(storyboard["subgoals"]))
                        if ((triggered | (not storyboard["subgoals"][i]['trigger']))&((not deleted) |(not storyboard["subgoals"][i]['delete'])))]
                
                textsurfs = [(storyfont.render(s, False, (255,255,255)),i)for (s,i) in text]
                heightenum = [textsurf.get_height()for (textsurf,i) in textsurfs]
                heightsum = sum(heightenum)
                tmpsum = 0

                for (s,i) in textsurfs:
                    if accomplished[i]:
                       s = storyfont.render(storyboard["subgoals"][i]["description"], False, (0,255,0))
                    screen.blit(s, (10, HEIGHT - 30 - heightsum+tmpsum))
                    tmpsum += s.get_height()
                textsurf = storyfont.render(storyboard["story"], False, (255, 255, 255))
                screen.blit(textsurf, (10, HEIGHT - 20))

            pygame.display.flip()

    cl.close()


if __name__ == '__main__':
    main()
