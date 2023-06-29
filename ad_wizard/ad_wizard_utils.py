


import pygame
from math import sqrt, fmod, floor
import numpy as np
argmin = np.argmin
radians = np.radians
sin = np.sin
cos = np.cos

import socket 
import time

DEBUG=False

class Button:
    def __init__(self, txt, font, cx, cy, w, h, color=(255,255,255)):
        self.font = font
        self.left = cx - w//2
        self.right = cx + w//2
        self.top = cy - h//2
        self.bottom = cy + h//2
        self.w = w
        self.h = h
        self.color = color

        self.set_text(txt)
        self.tx = self.left + (self.w - self.textsurf.get_width())//2
        self.ty = self.top + (self.h - self.textsurf.get_height())//2

    def set_text(self, txt):
        self.txt = txt
        self.textsurf = self.font.render(txt, True, self.color)

    def render(self, surf):
        pygame.draw.rect(surf, self.color, pygame.Rect(self.left, self.top, self.w, self.h), 3)
        surf.blit(self.textsurf, (self.tx, self.ty))

    def click(self, pos):
        return self.left <= pos[0] <= self.right and self.top <= pos[1] <= self.bottom


def transform2M(pitch, yaw, roll):
    Rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
    Ry = np.array([[cos(pitch), 0, -sin(pitch)], [0, 1, 0], [sin(pitch), 0, cos(pitch)]])
    Rx = np.array([[1, 0, 0], [0, cos(roll), sin(roll)], [0, -sin(roll), cos(roll)]])
    R = Rz.dot(Ry).dot(Rx)
    return R


def mToE(m):
    P1 = np.arcsin(m[2,0])
    P2 = np.pi - P1

    cosp1 = np.cos(P1)
    Y1 = np.arctan2(m[1,0]/cosp1, m[0,0]/cosp1)
    R1 = np.arctan2(-m[2,1]/cosp1, m[2,2]/cosp1)

    cosp2 = np.cos(P2)
    Y2 = np.arctan2(m[1,0]/cosp2, m[0,0]/cosp2)
    R2 = np.arctan2(-m[2,1]/cosp2, m[2,2]/cosp2)

    if np.sum((transform2M(P1,Y1,R1) - m)**2) < np.sum((transform2M(P2,Y2,R2) - m)**2):
        return (P1, Y1, R1)
    else:
        return (P2, Y2, R2)


def messageClient(host, port, inbox, outbox):
    textIn = ''

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        sock.setblocking(False)

        while 1:
            try:
                textIn += str(sock.recv(1024), "utf-8", errors='ignore')

                if '\n' in textIn:
                    endpos = textIn.index('\n')
                    msg = textIn[:endpos].strip()

                    if msg != 'keepalive':
                        inbox.put(msg)
                    textIn = textIn[endpos + 1:]

            except BlockingIOError:
                pass

            if not outbox.empty():
                msg = outbox.get()
                if DEBUG: print('Sending: %s' % msg)
                sock.sendall(bytes(msg.strip() + '\n', "utf-8"))
                # with open(logfn, 'a') as f:
                #    f.write('%.3f, "ad_wizard_msg", "%s"\n' % (time.time(), msg))
            time.sleep(0.05)
