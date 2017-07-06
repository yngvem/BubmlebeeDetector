import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
import os
import time
from bee_camera import BeeCamera
from sense_hat import SenseHat
from collections import OrderedDict


space_tolerance = 0.95 # What storage load to stop recording at

R = np.array([255, 0, 0])
G = np.array([0, 255, 0])
B = np.array([0, 0, 255])
W = np.array([255, 255, 255])
Bk = np.array([0, 0, 0])
C = G+B
M = R+B
Y = (R + 0.7*G).astype(int)

def set_leds(colour_dict, sense_hat):
    """Sets the colour of the sense hat's led display

    Parameters
    ----------
    colour_dict : collections.OrderedDict
       Ordered dictionary where the key is no. of led lights
       to light and the value is what colour to use for the
       given LEDs.
    sense_hat : sense_hat.SenseHat
       Controller for the sense hat.
    """
    if sum(colour_dict) != 64:
        raise ValueError('Sum of colour_dict keys should be 64')

    colour_array = []
    for n, colour in colour_dict.iteritems():
        colour_array += n*[colour]

    sense_hat.set_pixels(colour_array)


def free_space():
    dev_info = os.statvfs('/')
    total_space = dev_info.f_blocks
    avail_space = dev_info.f_bavail
    return avail_space/float(total_space)
    

def show_fraction(fraction, sense_hat):
    green = np.floor(fraction*64).astype(int)
    red = 64-green
    colour_dict = OrderedDict()
    colour_dict[red] = R
    colour_dict[green] = G
    set_leds(colour_dict, sense_hat)


if __name__ == '__main__':
    sense = SenseHat()
    sense.set_pixels([R]*64)
    with BeeCamera() as camera:
        camera.storage = '/home/pi/Bumblebees'
        camera.set_event_time(pre_event_time=60, post_event_time=60)

        space = free_space()
        curr_time = time.time()
        while(space < space_tolerance):
            show_fraction(space, sense)
            prev_time = curr_time
            curr_time = time.time()
            if curr_time - prev_time < 0.5:
                time.sleep(0.5 - (curr_time - prev_time))

            camera.detect_bumblebee()
            space = free_space()

        
        
        
        
