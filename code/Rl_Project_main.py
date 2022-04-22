#%%
# IMPORT LIBRARIES

import glob
import os
import matplotlib.pyplot as plt
import sys
import json
import numpy as np
from RL_funcs import *
import Carla_funcs as cf
import gcg

try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


# %%
# SETUP RL VARIABLE

# K-shooting time horizon
H = 16

# Number of K-shoots
K = 5

# Steering actions as a percentage of ~70° 
action_space = np.array([-0.3, 0, 0.3])

# Vehicle speed in m/s (this is decomposed into an orthogonal cartesian coordinate system and only approximates horizontal velocities)
vehicle_speed = 5

# Simulator time steps
delta_t = 0.1


# %%
# INITIALIZE THE TF NETWORK MODEL

# Use these two lines to build a TF model from scratch
model = gcg.computation_graph(H)
print('GCG Built!')

# Use these two lines to load a previously built TF model (used for outages and troubleshooting)
# model = gcg.load_old_model('../models/model.tf')
# print('GCG Loaded!')


# %%
#  CARLA SIMULATOR SETUP

# Initialize a connection with the CARLA server and setup all CARLA related experimental parameters and supports
client, world, vehicle, camera, collision, orig_settings, image_queue, collision_queue = cf.setup(time_step = delta_t, img_x = 128, img_y = 72, speed=vehicle_speed)
    

# %%
# ITERATE THROUGH A PREDIFIEND SET OF EPISODES

# Define the number of episodes
#big_loop_counter = 4000
big_loop_counter = 2

# Target a collision free runtime of 10 minutes
step_max = int((10 * 60) / delta_t)

# Setup an object to hold the stacked image state
img_stack = None

# Setup some metrics
cum_steps = 0
cum_steps_per_ep = []
steps_per_ep = []

# Iterate through the episodes
for i in range(big_loop_counter):
    
    # Initialize and episodic dataset
    dataset_I = []
    dataset_a = []
    y_labels = []
    y_buffer = np.zeros(H)

    episode_done = False
    
    # Retrieve data from the CARLA simulator (resulting from a "tick", see Carla_funcs.py)
    world.get_snapshot().frame
    
    # Retrieve and image from the image queue.  
    img = image_queue.get(True, 1.0)
    
    # Navigate through the simulator (up to 10 minutes collision free)
    for step in range(step_max):
    
        # Setup objects for a single iteration
        img_stack = cf.preprocess_img(img, img_stack)
        rwd_list = []
        action_sets = []
    
        # Do K-shooting
        for k in range(K):
            
            # Initialize a new action vector
            action_input = generate_actions(action_space, H)
            
            # Predict the probabilities of collision
            y_hats = gcg.run(model, img_stack, action_input)
            
            # Calculate the reward
            rwd = reward_function(y_hats)
            rwd_list.append(rwd)
            action_sets.append(action_input)
    
        # Extract the highest reward and action from the K-shoot
        best_action_set = best_actions(action_sets, rwd_list)
        
        # Take a single action in CARLA Simulator and observe the new state and "reward" signal 
        img, collided = cf.take_action(world, vehicle, image_queue, img, collision_queue, best_action_set[0])
        dataset_I.append(img_stack)
        dataset_a.append(best_action_set)
        
        # Populate actual collision labels
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
        elif step+1 >= H:
            y_labels.append(y_buffer.copy())

    cum_steps += step
    cum_steps_per_ep.append(cum_steps)
    steps_per_ep.append(step)

    # Update/train the model at the end of every episode
    gcg.train(model, dataset_I, dataset_a, y_labels)
    
    # Outputs for observing training, saving the model and storing metrics
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

    # Clean-up the CARLA server and get it read for the next episode
    cf.close(world, camera, collision, vehicle, orig_settings)
    vehicle, camera, collision, image_queue, collision_queue = cf.spawn_car(world= world, img_x = 128, img_y = 72, speed=vehicle_speed)



# %%
# SHUT DOWN CARLA

# Leave the CARLA Server in a clean state
cf.close(world, camera, collision, vehicle, orig_settings)


