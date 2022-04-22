#%%
# Define all RL functions

import numpy as np
import random

# Calculate the reward based on an vector of collision probabilities
def reward_function(col_probs):
    rwd = np.sum(-1*col_probs)
    return rwd

# Generate a list of random rewards
def generate_actions(action_space, num_actions):
    actions =[]
    for i in range(num_actions):
        action = random.choice(action_space)
        actions.append(action)
    actions=np.array(actions)
    return actions

# Return the action set that resulted in the best reward
def best_actions(action_sets, rwds):
    best = np.argmax(rwds)
    action = action_sets[best]
    return action

# Calculate a probability of collision
def future_prob_of_colision(y):
    b = np.mean(y)
    # bs = []
    # for y_hat in y_hats:
    #     b = np.mean(y_hat)
    #     bs.append(b)
    # b = np.min(bs)
    return b
