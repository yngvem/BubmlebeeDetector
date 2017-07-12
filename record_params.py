import numpy as np


# Normal mode if false, record test mode if True.
record_test_footage = False 

# How long default test footage record time is.
test_record_time = 180 

# How long to record before bumblebee was detected.
pre_event_time = 10

# How long to record after bumblebee was detected.
post_event_time = 10 

# What storage percentage to stop recording at.
space_tolerance = 0.95

# Where to save footage
footage_loc = '/home/pi/Bumblebees/'

# Sense hat colours
R = np.array([255, 0, 0])
G = np.array([0, 255, 0])
B = np.array([0, 0, 255])
W = np.array([255, 255, 255])
Bk = np.array([0, 0, 0])
C = G+B
M = R+B
Y = (R + 0.7*G).astype(int)
