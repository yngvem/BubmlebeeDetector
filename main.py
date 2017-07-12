import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
import os
import time
from pynput import keyboard
from bee_camera import BeeCamera
from sense_hat import SenseHat
from collections import OrderedDict
from record_params import *


sense = SenseHat()
test_mode = False

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


def record_test(key):
    global record_time
    
    if key == keyboard.Key.enter:
        if free_space() < space_tolerance:
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
        else:
            sense.show_message('Full SD card.')
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


def what_mode(key):
    global test_mode
    if key == keyboard.Key.left:
        sense.show_message('Test footage')
        test_mode = True
    elif key == keyboard.Key.right:
        sense.show_message('Normal mode')
        test_mode = True
    elif key == keyboard.Key.enter:
        sense.show_mesage('Starting')
        return False


if __name__ == '__main__':
    with keyboard.Listener(on_press=what_mode) as listener:
        listener.join()

    if test_mode == True:
        with keyboard.Listener(on_press=record_test) as listener:
            listener.join()

    else:
        sense.set_pixels([R]*64)
        with BeeCamera() as camera:
            camera.storage = footage_loc
            camera.set_event_time(pre_event_time=pre_event_time, post_event_time=post_event_time)

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
        