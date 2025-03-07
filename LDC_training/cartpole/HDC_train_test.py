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
from tensorflow import keras
from tensorflow.keras.models import load_model


#all the function needed
USE_CUDA = True # If we want to use GPU (powerful one needed!)
env = ContinuousCartPoleEnv()
GRAYSCALE = True # False is RGB
RESIZE_PIXELS = 60 # Downsample image to this number of pixels
FRAMES = 2 # state is the number of last frames: the more frames,
device = torch.device("cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu")
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

# LDC Ground truth
def load_LDC_model5(filename='sig16x16.yml'):
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
        tf.keras.layers.Dense(16, activation='sigmoid'),               # First hidden layer
        tf.keras.layers.Dense(16, activation='sigmoid'),               # Second hidden layer
        tf.keras.layers.Dense(16, activation='sigmoid'),               # third hidden layer
        tf.keras.layers.Dense(16, activation='sigmoid'),               # Fourth hidden layer
        tf.keras.layers.Dense(1, activation='tanh')                    # Output layer
    ])

    # Set the weights for each layer in the model
    model.layers[0].set_weights([weights[1].transpose(), biases[1]])
    model.layers[1].set_weights([weights[2].transpose(), biases[2]])
    model.layers[2].set_weights([weights[3].transpose(), biases[3]])
    model.layers[3].set_weights([weights[4].transpose(), biases[4]])
    model.layers[4].set_weights([weights[5].transpose(), biases[5]])
    # Compile the model with Adam optimizer and MSE loss
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

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



# HDC Ground truth
# initial setting for recorded set
env = ContinuousCartPoleEnv()

def Training_data_for_HDC(LDC):
    image_set = []; action_set = []; dot_set = [];image_set_np = []

    test_step = 20; test_point = 1
    pos_range = [0, 0.1]; vel_range = [0, 0.1]; theta_range = [0.05, 0.16]; dot_range = [-0.4, -0.35]
    big_split = 11; small_split = 6; Big_split = 12

    pos_cell = np.linspace(pos_range[0], pos_range[1], big_split)
    vel_cell = np.linspace(vel_range[0], vel_range[1], big_split)
    theta_cell = np.linspace(theta_range[0], theta_range[1], Big_split)
    dot_cell = np.linspace(dot_range[0], dot_range[1], small_split)
    start_time = time.time()
    for a in range(len(pos_cell)-1):
        for b in range(len(vel_cell)-1):
            for c in range(len(theta_cell)-1):
               for d in range(len(dot_cell)-1):
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
                        init_screen = get_screen()
                        screens = deque([init_screen] * FRAMES, FRAMES)
                        # resize the image
                        image4train = screens[1]
                        image4train = image4train.squeeze()
                        image4train = image4train.unsqueeze(-1)
                        image_set.append(image4train)

                        # resize the image
                        # image_input = screens[1]
                        # image_input = image_input.squeeze()
                        # image_input = image_input.unsqueeze(-1)
                        # image_input_np = [tensor.cpu().numpy() for tensor in image_input]
                        # image_input_np = np.stack(image_input_np, axis=0)
                        # image_input_np = np.expand_dims(image_input_np, axis=0)


                        dtheta = np.array([env.state[3]]); test_vel = np.random.rand(1)
                        dot_set.append(dtheta)

                        current_state = np.reshape(env.state, (-1, 4))
                        action = LDC.predict(current_state)
                        action = np.squeeze(action)  # switch the action to (1,) format
                        action = np.array([action], dtype=np.float32)
                        action_set.append(action)
                        # switch the action to (1,) format
                        # update the state
                        state_variables, _, done, __ = env.step(action)
                        env.render()

                    # for tensor in image_set:
                    #     if isinstance(tensor, torch.Tensor):  # Check if it's a tensor
                    #         if tensor.is_cuda:  # Check if the tensor is on CUDA
                    #             tensor = tensor.cpu()  # Move to CPU
                    #         if tensor.requires_grad:
                    #             tensor = tensor.detach()  # Detach from the computation graph if necessary
                    #         image_set_np.append(tensor.numpy())  # Convert to NumPy
                    #     else:
                    #         print("Non-tensor element encountered:", tensor)
                    # image_set_np = [tensor.cpu().numpy() for tensor in image_set]
                    # image_set_np = np.stack(image_set_np, axis=0)
                    # np.save('image_train_HDC_pos{0}.npy'.format(a), image_set_np)
                    # np.save('dtheta_train_HDC_pos{0}.npy'.format(a), dot_set)
                    # np.save('action_train_HDC_pos{0}.npy'.format(a), action_set)

    env.close()
    end_time = time.time()
    simulation_time = end_time - start_time


    image_set_np = [tensor.cpu().numpy() for tensor in image_set]
    image_set_np = np.stack(image_set_np, axis=0)
    np.save('image_train_HDC.npy', image_set_np)
    np.save('dtheta_train_HDC.npy', dot_set)
    np.save('action_train_HDC.npy', action_set)

    return image_set, dot_set, action_set, simulation_time


# LDC_whole = load_LDC_model('Cart_LDC1_whole_spce.yml')
# image_set, dot_set, action_set, simulation_time= Training_data_for_HDC(LDC_whole)


def HDC_model():
    # image_input = layers.Input(shape=(64, 64, 1), name="image_input") # assuming grayscale image
    image_input = layers.Input(shape=(96, 96, 1), name="image_input") # assuming grayscale image
    # CNN layers for image processing
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    # Velocity input
    velocity_input = layers.Input(shape=(1,), name="velocity_input")
    # Concatenate the CNN output and velocity
    combined = layers.Concatenate()([x, velocity_input])
    # Fully connected layers
    combined = layers.Dense(64, activation='relu')(combined)
    combined = layers.Dense(32, activation='relu')(combined)
    # Output layer
    action_output = layers.Dense(1, name="action_output")(combined)
    # Construct the model
    model = Model(inputs=[image_input, velocity_input], outputs=action_output)
    model.compile(optimizer='adam', loss='mean_squared_error') # assuming regression problem for action output
    return model

#HDC_architect = HDC_model()
# HDC_architect.summary() # to view the architecture

def train_HDC(HDC_architect):
    training_in_image = np.load('image_train_HDC.npy')
    training_out_action = np.load('action_train_HDC.npy')
    training_in_velocity = np.load('dtheta_train_HDC.npy')

    # Train the model
    history = HDC_architect.fit(
        [training_in_image, training_in_velocity],  # Input data
        training_out_action,                        # Output data
        epochs=10,                                  # Number of epochs; adjust based on your needs
        batch_size=16,                              # Batch size; adjust based on your needs
        validation_split=0.15                        # Use 20% of the data for validation; adjust based on your needs
    )
    return HDC_architect

# HDC_architect = HDC_model()
# HDC_architect.summary() # to view the architecture
# HDC_trained = train_HDC(HDC_architect)
# HDC_trained.save('trained_HDC_cartpole.h5')



#---------get training data for LDC------------
def get_training_4LDC(pos_start, pos_end, vel_start, vel_end, theta_start, theta_end, dot_start, dot_end, points, HDC, steps):
    HDC_model = HDC
    num_position = 2; num_theta = 2
    num_velocity = 2; num_dot = 2
    num_point = points; steps = steps
    positon_CP = np.linspace(pos_start, pos_end, num_position)
    velocity_CP = np.linspace(vel_start, vel_end, num_velocity)
    theta_CP = np.linspace(theta_start, theta_end, num_theta)
    dot_CP = np.linspace(dot_start, dot_end, num_dot)

    for i in range(num_position - 1):
        for j in range(num_velocity - 1):

            random_start_position = positon_CP[i];random_end_position = positon_CP[i + 1]
            random_start_velocity = velocity_CP[j]; random_end_velocity = velocity_CP[j + 1]
            random_start_theta = theta_CP[i]; random_end_theta = theta_CP[i + 1]
            random_start_dot = dot_CP[i]; random_end_dot = dot_CP[i + 1]

            random_positions = np.random.uniform(random_start_position, random_end_position, num_point)
            random_velocities = np.random.uniform(random_start_velocity, random_end_velocity, num_point)
            random_theta = np.random.uniform(random_start_theta, random_end_theta, num_point)
            random_dot = np.random.uniform(random_start_dot, random_end_dot, num_point)

            sampled_states = np.stack((random_positions, random_velocities, random_theta, random_dot), axis=-1)

            states_HDC = []
            action_HDC_array = []
            for k in range(num_point):
                # calculate the trajectory and action by HDC
                int_state = sampled_states[k]
                states_HDC.append(int_state)
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
                    states_HDC.append(next_state.astype(np.float32))
                    env.render()
                states_HDC.pop()

            np.save('training_input_LDC.npy', states_HDC)
            np.save('training_output_LDC.npy', action_HDC_array)
    return states_HDC, action_HDC_array

# HDC_model = tf.keras.models.load_model('trained_HDC_cartpole.h5')
# pos_start = 0; pos_end = 0.1; vel_start = 0; vel_end = 0.1; theta_start= 0.06; theta_end = 0.16; dot_start = -0.4; dot_end = -0.35
# points = 600; steps = 20
# states_HDC, action_HDC_array = get_training_4LDC(pos_start, pos_end, vel_start, vel_end, theta_start, theta_end, dot_start, dot_end ,  points, HDC_model, steps)



def HDC_GT(HDC_model):
    #initial try: but too long for verification
    # pos_range = [0, 0.1]; vel_range = [0, 0.1]; theta_range = [0.15, 0.25]; dot_range = [0, 0.1]
    # second try
    # pos_range = [0, 0.1]; vel_range = [0, 0.05]; theta_range = [0.15, 0.25]; dot_range = [0.05, 0.1]
    # num_split = 11; small_split = 6
    safe_point = []
    unsafe_point = []
    test_step = 20; test_point = 1
    pos_range = [0, 0.1]; vel_range = [0, 0.1]; theta_range = [0.06, 0.12]; dot_range = [-0.4, -0.35]
    num_split = 11; dot_split = 6; theta_split = 7

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
                        # # Old version is not accurate
                        # init_screen = get_screen()
                        # screens = deque([init_screen] * FRAMES, FRAMES)
                        # image_input = screens[1]
                        # image_input = image_input.squeeze()
                        # image_input = image_input.unsqueeze(-1)
                        # image_input_np = [tensor.cpu().numpy() for tensor in image_input]
                        # image_input_np = np.stack(image_input_np, axis=0)
                        # image_input_np = np.expand_dims(image_input_np, axis=0)
                        # dtheta = np.array([env.state[3]]);
                        # action_HDC = HDC_model.predict([image_input_np, dtheta])
                        # action_HDC = np.squeeze(action_HDC)  # switch the action to (1,) format
                        # action_HDC = np.array([action_HDC], dtype=np.float32)

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

                record_last_theta = np.array(record_last_theta)
                result = (record_last_theta >= -0.2) & (record_last_theta <= 0.2)
                if np.all(result):
                    safe_point.append(np.array([pos_cell[a], vel_cell[b], theta_cell[c], dot_cell[d]]).squeeze())
                else:
                    unsafe_point.append(np.array([pos_cell[a], vel_cell[b], theta_cell[c], dot_cell[d]]).squeeze())
    env.close()
    end_time = time.time()
    simulation_time = end_time - start_time

    np.save('safe_point_HDC_newaction.npy', safe_point)
    np.save('unsafe_point_HDC_new_action.npy', unsafe_point)
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



# test with new action method
record_pos = []; record_rad = []
record_vel = []; record_angvel = []
record_action = []
safe_point = []; unsafe_point = []

test_step = 20; test_point = 1
env = ContinuousCartPoleEnv()
HDC_model = tf.keras.models.load_model('trained_HDC_cartpole.h5')
#

#initial test on the data
for i in range(test_point):
    rand_pos = np.random.uniform(low=0, high=0.01, size=(1, 1))
    rand_theta = np.random.uniform(low=0.1, high=0.11, size=(1, 1))
    rand_vel = np.random.uniform(low=0.04, high=0.05, size=(1, 1))
    rand_dtheta = np.random.uniform(low=-0.38, high=-0.37, size=(1, 1))
    test_state = [rand_pos, rand_vel, rand_theta, rand_dtheta]
    test_state = np.array(test_state).squeeze()
    init_state = test_state
    env.state = init_state
    for _ in range(test_step):
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
        record_action.append(action_HDC)
        position = state_variables[0]; record_pos.append(position)
        velocity = state_variables[1]; record_vel.append(velocity)
        angle = state_variables[2]; record_rad.append(angle)
        anglevelocity = state_variables[3]; record_angvel.append(anglevelocity)

    if state_variables[2] >= -0.2 and state_variables[2] <= 0.2:
        safe_point.append(init_state)
    else:
        unsafe_point.append(init_state)
env.close()
pos_safe = [arr[0] for arr in safe_point]; theta_safe = [arr[2] for arr in safe_point]
pos_unsafe = [arr[0] for arr in unsafe_point];theta_unsafe = [arr[2] for arr in unsafe_point]

plt.scatter(pos_safe, theta_safe, color = 'green')
plt.scatter(pos_unsafe, theta_unsafe, color = 'red')

plt.title('test the safety')
plt.xlabel('time steps')
plt.ylabel('Position')
# Display the plot
plt.show()



# Test for one point
# initial setting for recorded set
record_pos = []; record_rad = []
record_vel = []; record_angvel = []
record_action = []
safe_point = []; unsafe_point = []

test_step = 20; test_point = 1
env = ContinuousCartPoleEnv()
HDC_model = tf.keras.models.load_model('trained_HDC_cartpole.h5')
#

#initial test on the data
for i in range(test_point):
    rand_pos = np.random.uniform(low=0, high=0.01, size=(1, 1))
    rand_theta = np.random.uniform(low=0.1, high=0.11, size=(1, 1))
    rand_vel = np.random.uniform(low=0.04, high=0.05, size=(1, 1))
    rand_dtheta = np.random.uniform(low=-0.38, high=-0.37, size=(1, 1))
    test_state = [rand_pos, rand_vel, rand_theta, rand_dtheta]
    test_state = np.array(test_state).squeeze()
    init_state = test_state
    env.state = init_state
    for _ in range(test_step):
        # env.reset()
        init_screen = get_screen()
        # previous discrete action
        screens = deque([init_screen] * FRAMES, FRAMES)
        # resize the image
        image_input = screens[1]
        image_input = image_input.squeeze()
        image_input = image_input.unsqueeze(-1)
        image_input_np = [tensor.cpu().numpy() for tensor in image_input]
        image_input_np = np.stack(image_input_np, axis=0)
        image_input_np = np.expand_dims(image_input_np, axis=0)
        dtheta = np.array([env.state[3]]);
        action_HDC = HDC_model.predict([image_input_np, dtheta])
        action_HDC = np.squeeze(action_HDC)  # switch the action to (1,) format
        action_HDC = np.array([action_HDC], dtype=np.float32)

        # update the state
        state_variables, _, done, __ = env.step(action_HDC)
        env.render()
        record_action.append(action_HDC)
        position = state_variables[0]; record_pos.append(position)
        velocity = state_variables[1]; record_vel.append(velocity)
        angle = state_variables[2]; record_rad.append(angle)
        anglevelocity = state_variables[3]; record_angvel.append(anglevelocity)

    if state_variables[2] >= -0.2 and state_variables[2] <= 0.2:
        safe_point.append(init_state)
    else:
        unsafe_point.append(init_state)
env.close()
pos_safe = [arr[0] for arr in safe_point]; theta_safe = [arr[2] for arr in safe_point]
pos_unsafe = [arr[0] for arr in unsafe_point];theta_unsafe = [arr[2] for arr in unsafe_point]

plt.scatter(pos_safe, theta_safe, color = 'green')
plt.scatter(pos_unsafe, theta_unsafe, color = 'red')

plt.title('test the safety')
plt.xlabel('time steps')
plt.ylabel('Position')
# Display the plot
plt.show()