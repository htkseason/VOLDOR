import numpy as np
from pylab import box
import cv2
import sys
import argparse


__all__ = ['load_flow', 'save_flow', 'vis_flow']

def load_flow(path):
    with open(path, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
        return None

def save_flow(path, flow):
    magic = np.array([202021.25], np.float32)
    h, w = flow.shape[:2]
    h, w = np.array([h], np.int32), np.array([w], np.int32)

    with open(path, 'wb') as f:
        magic.tofile(f); w.tofile(f); h.tofile(f); flow.tofile(f)

def vis_flow(flow, scale=0):
   fx, fy = cv2.split(flow)
   mag,ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
   if scale== 0:
      cv2.normalize(mag, mag, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
   else:
      mag /= scale
   ret = cv2.merge([ang,mag,np.ones_like(mag)])
   ret = cv2.cvtColor(ret, cv2.COLOR_HSV2BGR)
   return ret
