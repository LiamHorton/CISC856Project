import numpy as np
import random

def cost_function(col_probs, b):
    cost = np.sum(-1*col_probs)-b
    return cost

def generate_actions(action_space, num_actions):
    actions = np.empty([1,num_actions])
    for i in range(num_actions):
        action = random.choices(action_space)
        actions[i] = action
    return actions

def best_actions(action_sets, costs):
    best = np.argmax(costs)
    action = action_sets[best]
    return action

def future_prob_of_colision(y):
    b = np.mean(y)
    # bs = []
    # for y_hat in y_hats:
    #     b = np.mean(y_hat)
    #     bs.append(b)
    # b = np.min(bs)
    return b

def create_y_label(y_buffer, y):
