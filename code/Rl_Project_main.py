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
import gcg


# %%
# RL variables
H = 8
K = 5
action_space = np.array([-0.05, 0, 0.05])

# %%
# Initialize Network

model = gcg.computation_graph()

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
    cum_steps = 0
    cum_steps_per_ep = []
    steps_per_ep = []
    # Sim loop
    episode_done = False
    # get an initial image
    world.get_snapshot().frame
    img = image_queue.get(True, 1.0)
    for step in range(step_max):
        img_stack = cf.preprocess_img(img, img_stack)
        rwd_list = []
        action_sets = []
        for k in range(K):
            action_input = generate_actions(action_space, H)
            y_hats = gcg.run(model, img_stack, action_input)
            rwd = reward_function(y_hats)
            rwd_list.append(rwd)
            action_sets.append(action_input)
        best_action_set = best_actions(action_sets, rwd_list)
        img, collided = cf.take_action(world, vehicle, image_queue, collision_queue, best_actions[0])
        dataset_I.append(img_stack)
        dataset_a.append(best_action_set)
        if step<H:
            y_buffer[step] = collided
        elif step == H:
            y_labels.append(y_buffer)
        else:
            y_buffer[0:-2] = y_buffer[1:-1]
            y_buffer[-1] = collided
            y_labels.append(y_buffer)
        if collided == 1:
            break
    for i in range(H-1):
        y_buffer[0:-2] = y_buffer[1:-1]
        y_buffer[-1] = 1
        y_labels.append(y_buffer)
    cum_steps += step
    cum_steps_per_ep.append(step)
    steps_per_ep.append(step)

    gcg.train(model, dataset_I, dataset_a, y_labels)

# %%
# End and exit
carla.close(world, camera, collision, vehicle, orig_settings)