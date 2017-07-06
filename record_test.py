import numpy as np
import picamera
import time
import os
import io
import cv2
from pynput import keyboard
from bee_camera import BeeCamera
from sense_hat import SenseHat
from collections import OrderedDict

sense = SenseHat()
R = np.array([255, 0, 0])
G = np.array([0, 255, 0])
B = np.array([0, 0, 255])
W = np.array([255, 255, 255])
O = np.array([0, 0, 0])

record_time = 300

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



print('Setting up camera')
sense.set_pixels([B]*64)
camera = BeeCamera(
    resolution=(1280, 960),
    pre_event_time=record_time/2,
    post_event_time=record_time/2
)
space = free_space()
show_fraction(space, sense)
print('Camera ready')

def record(key):
    global record_time
    
    if key == keyboard.Key.enter:
        print('Recording test footage')
        sense.set_pixels([W]*64)
        camera.record_test_footage(sense)
        print('Finished recording')
    elif key == keyboard.Key.up:
        record_time += 10
        camera.set_event_time(
            pre_event_time=record_time/2,
            post_event_time=record_time/2
        )
        sense.show_message(str(record_time))
    elif key == keyboard.Key.down:
        record_time -= 10
        if record_time < 10:
            record_time = 10
        camera.set_event_time(
            pre_event_time=record_time/2,
            post_event_time=record_time/2
        )
        sense.show_message(str(record_time))
    elif key == keyboard.Key.left:
        record_time -= 60
        if record_time < 10:
            record_time = 10
        camera.set_event_time(
            pre_event_time=record_time/2,
            post_event_time=record_time/2
        )
        sense.show_message(str(record_time))
    elif key == keyboard.Key.right:
        record_time += 60
        camera.set_event_time(
            pre_event_time=record_time/2,
            post_event_time=record_time/2
        )
        sense.show_message(str(record_time))
    elif key == keyboard.Key.esc:
        sense.set_pixels([O]*64)
        return False
    space = free_space()
    show_fraction(space, sense)

with keyboard.Listener(on_press=record) as listener:
    listener.join()
    

