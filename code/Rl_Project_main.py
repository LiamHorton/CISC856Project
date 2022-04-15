
import glob
import os
from pickle import TRUE
from queue import Queue
import matplotlib.pyplot as plt
import sys
import cv2
from zmq import QUEUE

try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
from queue import Queue
from queue import Empty
from PIL import Image
import numpy as np
from RL_funcs import *
import Carla_funcs as cf



# %%
# RL variables
H = 8
K = 5
action_space = np.array([-0.05, 0, 0.05])

# %%
# Initialize Network

GCG = gcg.model_initialize()

# %%
#  Carla Setup
image_queue = Queue()
collision_queue = Queue()

client, world, vehicle, camera, collision, orig_settings = cf.setup(collision_queue, image_queue, time_step = 0.1, img_x = 300, img_y = 400)

# %%
# Begin loop
big_loop_counter = 5
step_max = 300000
img_stack = None
for i in range(big_loop_counter):
    # Initialize datasets
    dataset_I = []
    dataset_a = []
    y_labels = []
    y_buffer = np.empty([1,H])
    # Sim loop
    episode_done = False
    img, _ = cf.take_action(world, vehicle, image_queue, collision_queue, 0, speed=0)
    for step in range(step_max):
        img_stack = cf.preprocess(img, img_stack)
        cost_list = []
        action_sets = []
        for k in range(K):
            action_input = generate_actions(action_space, H)
            y_hats, b_hat = GCG.run([img_stack, action_input])
            cost = cost_function(y_hats, b_hat)
            cost_list.append(cost)
            action_sets.append(action_input)
        best_action_set = best_actions(action_sets, cost_list)
        img, collided = cf.take_action(world, vehicle, image_queue, collision_queue, best_actions[0])
        dataset_I.append(img_stack)
        dataset_a.append(best_action_set)
        if step<H:
            y_buffer[step] = collided
        else:









# %%
# End and exit
carla.close(world, camera, collision, vehicle, orig_settings)