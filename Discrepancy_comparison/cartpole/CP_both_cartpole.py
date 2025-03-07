import time

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque
from tqdm import tqdm
import tkinter
from continuous_cartpole import ContinuousCartPoleEnv
import gym
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model



#all the function needed
USE_CUDA = True # If we want to use GPU (powerful one needed!)
env = ContinuousCartPoleEnv()
GRAYSCALE = True # False is RGB
RESIZE_PIXELS = 60 # Downsample image to this number of pixels
FRAMES = 2 # state is the number of last frames: the more frames,
device = torch.device("cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu")


def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    # screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    img = env.render(mode='rgb_array')

    new_image_array = np.ones((600, 600, 3), dtype=np.uint8) * 255  # 255 represents white

    new_image_array[100:500, :, :] = img

    resized_image_array = cv2.resize(new_image_array, (96, 96))
    resized_image_array[63:64, :, :] = [0, 0, 0]  # Set pixel values to zero (black)
    # screen = resized_image_array
    # resized_screen = cv2.resize(screen, (135, 135))
    # screen = resized_screen[35:95]
    resized_image_array = cv2.cvtColor(resized_image_array, cv2.COLOR_BGR2GRAY)
    # screen = np.ascontiguousarray(gray_image, dtype=np.float32) / 255
    # screen = torch.from_numpy(screen)
    final = torch.tensor(resized_image_array/255).view(1,1,96,96).to(device).float()
    # Resize, and add a batch dimension (BCHW)
    return final


def load_LDC_model(filename='sig16x16.yml'):
    # Load the neural network parameters from the YAML file
    with open(filename, 'r') as file:
        nn_params = yaml.load(file, Loader=yaml.FullLoader)

    # Extract weights, biases, and activations
    weights = nn_params['weights']
    biases = nn_params['offsets']

    # Convert lists to numpy arrays with correct shapes
    for layer_index in weights.keys():
        weights[layer_index] = np.array(weights[layer_index])
        biases[layer_index] = np.array(biases[layer_index])

    # Construct the model using TensorFlow and Keras
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(4,), activation='sigmoid'),  # Input layer
        tf.keras.layers.Dense(16, activation='sigmoid'),                   # First hidden layer
        tf.keras.layers.Dense(1, activation='tanh')                        # Output layer
    ])

    # Set the weights for each layer in the model
    model.layers[0].set_weights([weights[1].transpose(), biases[1]])
    model.layers[1].set_weights([weights[2].transpose(), biases[2]])
    model.layers[2].set_weights([weights[3].transpose(), biases[3]])

    # Compile the model with Adam optimizer and MSE loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model






# calculate the CP over the whole state space with 1 LDC;
def get_two_dis(pos_start, pos_end, vel_start, vel_end, theta_start, theta_end, dot_start, dot_end ,  points, HDC, LDC, steps):
    HDC_model = HDC
    LDC_model = LDC # load the yml file
    num_position = 2; num_theta = 2
    num_velocity = 2; num_dot  = 2
    num_point = points
    steps = steps
    positon_CP = np.linspace(pos_start, pos_end, num_position)
    velocity_CP = np.linspace(vel_start, vel_end, num_velocity)
    theta_CP = np.linspace(theta_start, theta_end, num_theta)
    dot_CP = np.linspace(dot_start, dot_end, num_dot)

    for i in range(num_position - 1):
        for j in range(num_velocity - 1):
            random_start_position = positon_CP[i];random_end_position = positon_CP[i + 1]
            random_start_velocity = velocity_CP[j]; random_end_velocity = velocity_CP[j + 1]
            random_start_theta = theta_CP[i]; random_end_theta = theta_CP[i+1]
            random_start_dot = dot_CP[i]; random_end_dot = dot_CP[i+1]

            random_positions = np.random.uniform(random_start_position, random_end_position, num_point)
            random_velocities = np.random.uniform(random_start_velocity, random_end_velocity, num_point)
            random_theta = np.random.uniform(random_start_theta, random_end_theta, num_point)
            random_dot = np.random.uniform(random_start_dot, random_end_dot, num_point)

            sampled_states = np.stack((random_positions, random_velocities, random_theta, random_dot), axis=-1)

            max_diff_traj = []
            max_diff_action = []
            for k in range(num_point):
                states_LDC = []
                action_LDC_array = []
                # calculate the trajectory and action by LDC
                states_LDC.append(sampled_states[k][0])
                env.state = sampled_states[k]
                for m in range(steps):  # simulation for the LDC
                    current_state = np.reshape(env.state, (-1, 4))
                    action_LDC = LDC_model.predict(current_state)
                    action_LDC = np.squeeze(action_LDC)  # switch the action to (1,) format
                    action_LDC = np.array([action_LDC], dtype=np.float32)
                    # save the control action
                    action_LDC_array.append(action_LDC)

                    next_state_LDC, reward, done, _ = env.step(action_LDC)
                    env.render()
                    states_LDC.append(next_state_LDC[2])

                # calculate the trajectory and action by HDC
                states_HDC = []
                action_HDC_array = []
                int_state = sampled_states[k]
                states_HDC.append(int_state[0])
                env.state = sampled_states[k]
                for o in range(steps):
                    test_image = get_screen()
                    test_image = test_image.squeeze()
                    test_image = test_image.unsqueeze(-1)
                    test_image_np = [tensor.cpu().numpy() for tensor in test_image]
                    test_image_np = np.stack(test_image_np, axis=0)
                    test_vel = np.array([env.state[2]])
                    test_image_np = np.expand_dims(test_image_np, axis=0)
                    action_HDC = HDC_model.predict([test_image_np, test_vel])
                    action_HDC = np.squeeze(action_HDC)  # switch the action to (1,) format
                    action_HDC = np.array([action_HDC], dtype=np.float32)


                    action_HDC_array.append(action_HDC)
                    next_state, reward, done, _ = env.step(action_HDC)
                    states_HDC.append(next_state[2].astype(np.float32))
                    env.render()


                one_trajectory_max_diff = []
                for q in range(len(states_LDC)):
                    one_trajectory_max_diff.append(abs(states_LDC[q] - states_HDC[q]))

                one_action_max_diff = []
                for p in range(len(action_LDC_array)):
                    one_action_max_diff.append(abs(action_LDC_array[p] - action_HDC_array[p]))

                max_diff_action.append(max(one_action_max_diff))
                max_diff_traj.append(max(one_trajectory_max_diff))

    index = int(points * 0.04)
    sorted_traj_CP = sorted(max_diff_traj)
    traj_CP = sorted_traj_CP[-index]
    # CP value for the action-based
    index = int(points * 0.04)
    sorted_action_CP_all = sorted(max_diff_action)
    act_CP = sorted_action_CP_all[-index]

    return traj_CP, act_CP, max_diff_traj, max_diff_action

LDC_whole = load_LDC_model('Cart_LDC1_3layers_new.yml')
HDC_model = tf.keras.models.load_model('trained_HDC_cartpole.h5')
pos_start = 0; pos_end = 0.1; vel_start = 0; vel_end = 0.1; theta_start= 0.06; theta_end = 0.16; dot_start = -0.4; dot_end = -0.35
points = 60; steps = 20
traj_CP, act_CP, max_diff_traj, max_diff_action = get_two_dis(pos_start, pos_end, vel_start, vel_end, theta_start, theta_end, dot_start, dot_end ,  points, HDC_model, LDC_whole, steps)





# calculate the CP with LDCs.


# calculate the CP for small regions.