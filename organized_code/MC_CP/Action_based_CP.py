import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import gym
import cv2
import yaml
from tensorflow.keras.models import load_model
import pandas as pd


def load_LDC_model(filename):
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
        tf.keras.layers.Dense(16, input_shape=(2,), activation='sigmoid'),  # Input layer
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

def process_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (64, 64))
    image = np.float32(np.true_divide(image, 255))
    return image





#def action_CP_table():
# env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
# HDC_model = tf.keras.models.load_model('trained_HDC_cnn_model.h5')
#
# LDC_model = load_LDC_model('MC_LDCs/LDC2_(0,0.07).yml') # load the yml file
# #LDC_model = load_model('LDC1_whole_space.h5')
# num_position = 4
# num_velocity = 7
# num_point = 100
# positon_CP = np.linspace(-1.2, 0.6, num_position)
# velocity_CP = np.linspace(0, 0.07, num_velocity)
#
#
#
#
# CP_table = [[0 for _ in range(num_position-1)] for _ in range(num_velocity-1)]
#
# for i in range(num_position - 1):
#     for j in range(num_velocity - 1):
#         random_start_position = positon_CP[i]
#         random_end_position = positon_CP[i + 1]
#         random_start_velocity = velocity_CP[j]
#         random_end_velocity = velocity_CP[j + 1]
#
#         random_positions = np.random.uniform(random_start_position, random_end_position, num_point)
#         random_velocities = np.random.uniform(random_start_velocity, random_end_velocity, num_point)
#         sampled_states = np.stack((random_positions, random_velocities), axis=-1)
#
#         action_set = []
#         diff_set = []
#         for k in range(num_point):
#             desired_state = sampled_states[k]
#             env.reset(specific_state=desired_state)
#             state_array = desired_state.reshape(1, -1)
#             action_LDC = LDC_model.predict(state_array)
#             action_LDC = action_LDC[0]
#
#             image = env.render()
#             frame = process_image(image)
#             velocity = env.state[1]
#             frame = np.reshape(frame, (1, 64, 64, 1))
#             velocity = np.reshape(velocity, (1, 1))
#             action_HDC = HDC_model.predict([frame, velocity])
#             diff = abs(action_HDC - action_LDC)
#             diff_set.append(diff)
#
#         diff_sort = sorted(diff_set)
#
#         index = int(len(diff_sort) * 0.04)
#         # Get the top 4% values, which is equivalent to the top 96% values
#         top_96_percent_values = diff_sort[-index:]
#         top_96_percent = top_96_percent_values[0]
#         CP_table[j][i] = top_96_percent
#


# Action_based_CP for the Whole State Space.
env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
HDC_model = tf.keras.models.load_model('trained_HDC_cnn_model.h5')
#LDC_model = load_LDC_model() # load the yml file
#LDC_model = load_model('LDC1_whole_space.h5')
LDC_model = load_LDC_model(filename='MC_LDCs/LDC3_1st_(-0.6,0.6)(0,0.07).yml')

#this is for the whole state space
lower_bounds = env.observation_space.low
upper_bounds = env.observation_space.high
position_bounds = [lower_bounds[0], upper_bounds[0]]
velocity_bounds = [lower_bounds[1], upper_bounds[1]]

#this is for LDC2
velocity_bounds = [0, upper_bounds[1]]

#this is for LDC3
position_bounds = [-0.6, 0.6]
velocity_bounds = [0, 0.07]

# Randomly sample position and velocity
num_samples = 100
random_positions = np.random.uniform(position_bounds[0], position_bounds[1], num_samples)
random_velocities = np.random.uniform(velocity_bounds[0], velocity_bounds[1], num_samples)

# Stack them together to get the sampled states
sampled_states = np.stack((random_positions, random_velocities), axis=-1)

action_set = []
diff_set = []
for i in range(num_samples):
    desired_state = sampled_states[i]
    env.reset(specific_state=desired_state)
    state_array = desired_state.reshape(1, -1)
    action_LDC = LDC_model.predict(state_array)
    action_LDC = action_LDC[0]

    image = env.render()
    frame = process_image(image)
    velocity = env.state[1]
    frame = np.reshape(frame, (1, 64, 64, 1))
    velocity = np.reshape(velocity, (1, 1))
    action_HDC = HDC_model.predict([frame, velocity])
    diff = abs(action_HDC - action_LDC)
    diff_set.append(diff)

diff_sort = sorted(diff_set)

index = int(len(diff_sort) * 0.04)

# Get the top 4% values, which is equivalent to the top 96% values
top_96_percent_values = diff_sort[-index:]
print("the CP vaule will be:", top_96_percent_values)


