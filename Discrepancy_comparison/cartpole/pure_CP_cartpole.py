
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
env = ContinuousCartPoleEnv()

def show(image):
    plt.imshow(image.transpose(1, 2, 0))  # Transpose to (400, 600, 3) for displaying
    plt.axis('on')  # Turn off axis
    plt.show()
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



def HDC_GT(HDC_model):
    #initial try: but too long for verification
    # pos_range = [0, 0.1]; vel_range = [0, 0.1]; theta_range = [0.15, 0.25]; dot_range = [0, 0.1]
    # second try
    # pos_range = [0, 0.1]; vel_range = [0, 0.05]; theta_range = [0.15, 0.25]; dot_range = [0.05, 0.1]
    # num_split = 11; small_split = 6
    safe_point = []
    unsafe_point = []
    test_step = 20; test_point = 20
    pos_range = [0, 0.1]; vel_range = [0, 0.1]; theta_range = [0.06, 0.12]; dot_range = [-0.4, -0.35]
    num_split = 6; dot_split = 6; theta_split = 4

    pos_cell = np.linspace(pos_range[0], pos_range[1], num_split)
    vel_cell = np.linspace(vel_range[0], vel_range[1], num_split)
    theta_cell = np.linspace(theta_range[0], theta_range[1], theta_split)
    dot_cell = np.linspace(dot_range[0], dot_range[1], dot_split)
    start_time = time.time()
    for a in range(len(pos_cell) - 1):
        for b in range(len(vel_cell) - 1):
            for c in range(len(theta_cell) - 1):
               for d in range(len(dot_cell) - 1):
                record_last_theta = []
                for i in range(test_point):
                    rand_pos = np.random.uniform(low=pos_cell[a], high=pos_cell[a+1], size=(1, 1))
                    rand_theta = np.random.uniform(low=theta_cell[c], high=theta_cell[c+1], size=(1, 1))
                    rand_vel = np.random.uniform(low=vel_cell[b], high=vel_cell[b+1], size=(1, 1))
                    rand_dtheta = np.random.uniform(low=dot_cell[d], high=dot_cell[d+1], size=(1, 1))

                    test_state = [rand_pos, rand_vel, rand_theta, rand_dtheta]
                    test_state = np.array(test_state).squeeze()
                    init_state = test_state
                    env.state = init_state
                    for _ in range(test_step):
                        # New control version
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

                         # update the state
                        state_variables, _, done, __ = env.step(action_HDC)
                        env.render()


                    record_last_theta.append(state_variables[2])

                sort_last_theta = sorted(record_last_theta)
                record_last_theta = sort_last_theta[-2]
                result = (record_last_theta >= -0.2) & (record_last_theta <= 0.2)
                if np.all(result):
                    pos_1 = pos_cell[a]; pos_2 = pos_cell[a] + 0.01; pos_cp = [pos_1, pos_2]
                    vel_1 = vel_cell[b]; vel_2 = vel_cell[b] + 0.01; vel_cp = [vel_1, vel_2]
                    theta_1 = theta_cell[c]; theta_2 = theta_cell[c] + 0.01; thata_cp = [theta_1, theta_2]
                    for u in range(2):
                        for h in range(2):
                            for b in range(2):
                                safe_point.append(np.array([pos_cp[u], vel_cp[h], thata_cp[b], dot_cell[d]]).squeeze())
                else:
                    pos_1 = pos_cell[a]; pos_2 = pos_cell[a] + 0.01; pos_cp = [pos_1, pos_2]
                    vel_1 = vel_cell[b]; vel_2 = vel_cell[b] + 0.01; vel_cp = [vel_1, vel_2]
                    theta_1 = theta_cell[c]; theta_2 = theta_cell[c] + 0.01; thata_cp = [theta_1, theta_2]
                    for u in range(2):
                        for h in range(2):
                            for b in range(2):
                                unsafe_point.append(np.array([pos_cp[u], vel_cp[h], thata_cp[b], dot_cell[d]]).squeeze())

    env.close()
    end_time = time.time()
    simulation_time = end_time - start_time

    np.save('safe_point_HDC_CP.npy', safe_point)
    np.save('unsafe_point_HDC_CP.npy', unsafe_point)
    return safe_point, unsafe_point, simulation_time

HDC_model = tf.keras.models.load_model('trained_HDC_cartpole.h5')
safe_point, unsafe_point, simulation_time = HDC_GT(HDC_model)
pos_safe = [arr[0] for arr in safe_point]; theta_safe = [arr[2] for arr in safe_point]
pos_unsafe = [arr[0] for arr in unsafe_point]; theta_unsafe = [arr[2] for arr in unsafe_point]
plt.scatter(pos_safe, theta_safe, color = 'green')
plt.scatter(pos_unsafe, theta_unsafe, color = 'red')

plt.title('test the safety')
plt.xlabel('time steps')
plt.ylabel('Position')
# Display the plot
plt.show()
