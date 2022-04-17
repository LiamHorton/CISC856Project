#%%

# IMPORT LIBRARIES

import glob
import os
from pickle import TRUE
from queue import Queue
from tkinter import Y
import matplotlib.pyplot as plt
import sys
import cv2
from zmq import QUEUE
import tensorflow as tf
import json

try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
#from queue import Queue
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
action_space = np.array([-0.2, 0, 0.2])
vehicle_speed = 5
delta_t = 0.1


# %%
# Initialize Network
# model = gcg.computation_graph(H)
# print('GCG Built!')
model = gcg.load_old_model('../models/model.tf')
print('GCG Loaded!')


# %%
#  Carla Setup
# image_queue = Queue()
# collision_queue = Queue()

client, world, vehicle, camera, collision, orig_settings, image_queue, collision_queue = cf.setup(time_step = delta_t, img_x = 128, img_y = 72, speed=vehicle_speed)
    

# %%

# Begin loop
big_loop_counter = 5000
step_max = int((10 * 60) / delta_t)
img_stack = None

cum_steps = 0
cum_steps_per_ep = []
steps_per_ep = []

for i in range(big_loop_counter):
    # Initialize datasets
    dataset_I = []
    dataset_a = []
    y_labels = []
    y_buffer = np.zeros(H)

    # Sim loop
    episode_done = False
    # get an initial image
    
    world.get_snapshot().frame
    img = image_queue.get(True, 1.0)
    
    for step in range(step_max):
    #for step in range(9):
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
        img, collided = cf.take_action(world, vehicle, image_queue, img, collision_queue, best_action_set[0])
        dataset_I.append(img_stack)
        dataset_a.append(best_action_set)
        
        if collided == 1:
            if step+1 >= H:

                for j in range(H):
                    y_buffer[H-1-j] = 1
                    y_labels.append(y_buffer.copy())
                break

            else:
                for j in range(step):
                    y_buffer[H-j-step:] = 1
                    y_labels.append(y_buffer.copy())
                break
        elif  step+1 >= H:
            y_labels.append(y_buffer.copy())

    cum_steps += step
    cum_steps_per_ep.append(cum_steps)
    steps_per_ep.append(step)


    gcg.train(model, dataset_I, dataset_a, y_labels)
    
    if (i % 10 == 0) and (i > 0):
        print('Big Loop iteration - ', i)
    
    if (i % 50 == 0)and (i > 0):
        model.save('../models/model.tf')
        print('Model saved')

        with open("../images/cum_steps.json", 'w') as f:
            # indent=2 is not needed but makes the file human-readable
            # if the data is nested
            json.dump(cum_steps_per_ep, f, indent=2)

        with open("../images/steps.json", 'w') as f:
            # indent=2 is not needed but makes the file human-readable
            # if the data is nested
            json.dump(steps_per_ep, f, indent=2)
        
        fig1 = plt.figure()
        plt.plot(cum_steps_per_ep)
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Moves')
        plt.title('Figure 1')
        plt.savefig('../images/figure1.png')


        fig2 = plt.figure()
        plt.plot(steps_per_ep)
        plt.xlabel('Episodes')
        plt.ylabel('Moves')
        plt.title('Figure 2')
        plt.savefig('../images/figure2.png')

    


    cf.close(world, camera, collision, vehicle, orig_settings)
    client, world, vehicle, camera, collision, orig_settings, image_queue, collision_queue = cf.setup(time_step = delta_t, img_x = 128, img_y = 72, speed=vehicle_speed)

# %%
# End and exit
world.apply_settings(orig_settings)
cf.close(world, camera, collision, vehicle, orig_settings)


