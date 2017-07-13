import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
import os
import time
import sys
from pynput import keyboard
from bee_camera import BeeCamera
from sense_hat import SenseHat
from collections import OrderedDict
from record_params import *



sense = SenseHat()


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
    global test_record_time
    
    if key == keyboard.Key.enter:
        if free_space() < space_tolerance:
            print('Recording test footage, should take {} seconds'
                  .format(camera.pre_event_time+camera.post_event_time))
            sense.set_pixels([W]*64)
            camera.record_test_footage(sense)
            print('Finished recording')
        else:
            sense.show_message('Full SD card.')
    elif key == keyboard.Key.up:
        test_record_time += 10
        camera.set_event_time(
            pre_event_time=test_record_time/2,
            post_event_time=test_record_time/2
        )
    elif key == keyboard.Key.down:
        test_record_time -= 10
        if test_record_time < 10:
            test_record_time = 10
            camera.set_event_time(
                pre_event_time=test_record_time/2,
                post_event_time=test_record_time/2
            )
    elif key == keyboard.Key.left:
        test_record_time -= 60
        if test_record_time < 10:
            test_record_time = 10
            camera.set_event_time(
                pre_event_time=test_record_time/2,
                post_event_time=test_record_time/2
            )
    elif key == keyboard.Key.right:
        test_record_time += 60
        camera.set_event_time(
            pre_event_time=test_record_time/2,
            post_event_time=test_record_time/2
        )
    elif key == keyboard.Key.esc:
        sense.set_pixels([Bk]*64)
        return False
    sense.show_message(str(test_record_time))
    print('Record time: {}'.format(test_record_time))
    space = free_space()
    show_fraction(space, sense)


def what_mode(key):
    global manual_mode
    if key == keyboard.Key.left:
        sense.show_message('M')
        print('Manual mode')
        manual_mode = True
    elif key == keyboard.Key.right:
        sense.show_message('N')
        print('Normal mode')
        manual_mode = False
    elif key == keyboard.Key.enter:
        sense.show_message(':)')
        return False
    elif key == keyboard.Key.esc:
        sys.exit()


if __name__ == '__main__':
    sense.set_pixels([Pu]*64)
    print('Choose recording mode')
    with keyboard.Listener(on_press=what_mode) as listener:
        listener.join()
    space = free_space()
    show_fraction(space, sense)
    with BeeCamera(threshold=blob_thresh, min_area=min_blob_size, max_area=max_blob_size) as camera:
        print('Camera loaded')
        camera.set_storage(footage_loc)
        if manual_mode:
            print('Manual mode')
            with keyboard.Listener(on_press=record_test) as listener:
                listener.join()
        else:
            print('Normal mode')
            sense.set_pixels([R]*64)
            camera.set_event_time(pre_event_time=pre_event_time, post_event_time=post_event_time)

            space = free_space()
            curr_time = time.time()
            ten_its_time = curr_time
            num_its = 0L
            while(space < space_tolerance):
                if num_its % 10 == 0:
                    print('Checked 10 frames, took {:02} seconds'.format(curr_time - ten_its_time))
                    ten_its_time = curr_time
                num_its += 1
                show_fraction(space, sense)
                prev_time = curr_time
                curr_time = time.time()
                if curr_time - prev_time < 0.5:
                    time.sleep(0.5 - (curr_time - prev_time))

                camera.detect_bumblebee()
                space = free_space()
        
