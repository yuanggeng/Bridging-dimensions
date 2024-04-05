import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import gym
import cv2
import yaml
from tensorflow.keras.models import load_model
import pandas as pd


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


env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env

HDC_model = tf.keras.models.load_model('trained_HDC_cnn_model.h5')

LDC_model = load_LDC_model('LDC_new_(-0.6,-0.4)(-0.02,0.07).yml') # load the yml file
#LDC_model = load_model('LDC1_whole_space.h5')
num_position = 2
num_velocity = 2
num_point = 60
steps = 100
positon_CP = np.linspace(-0.6, -0.4, num_position)
velocity_CP = np.linspace(-0.02, 0.07, num_velocity)

CP_table = [[0 for _ in range(num_position - 1)] for _ in range(num_velocity - 1)]
CP_action_table = [[0 for _ in range(num_position - 1)] for _ in range(num_velocity - 1)]


for i in range(num_position - 1):
    for j in range(num_velocity - 1):
        random_start_position = positon_CP[i]
        random_end_position = positon_CP[i + 1]
        random_start_velocity = velocity_CP[j]
        random_end_velocity = velocity_CP[j + 1]

        random_positions = np.random.uniform(random_start_position, random_end_position, num_point)
        random_velocities = np.random.uniform(random_start_velocity, random_end_velocity, num_point)
        sampled_states = np.stack((random_positions, random_velocities), axis=-1)

        action_set = []
        diff_set = []
        states_HDC = []
        states_LDC = []
        max_diff_traj = []
        max_diff_action = []
        for k in range(num_point):
            states_HDC = []
            states_LDC = []
            action_LDC_array = []
            #calculate the trajectory by LDC
            states_LDC.append(sampled_states[k][0])
            desired_state = sampled_states[k]
            for m in range(steps): #simulation for the LDC
                env.reset(specific_state=desired_state)
                state_array = desired_state.reshape(1, -1)

                action_LDC = LDC_model.predict(state_array)
                action_LDC = action_LDC[0]
                action_LDC_array.append(action_LDC)
                next_state_LDC, reward, done, _, _ = env.step(action_LDC)
                states_LDC.append(next_state_LDC[0])
                desired_state = next_state_LDC


            #calculate the trajectory by HDC
            states_HDC = []
            action_HDC_array = []
            int_state = sampled_states[k]
            states_HDC.append(int_state[0])

            env.reset(specific_state=int_state)
            image = env.render()
            frame = process_image(image)
            vel_input = sampled_states[k][1] # the velocity should be the second one.
            frame = np.reshape(frame, (1, 64, 64, 1))
            vel_input = np.reshape(vel_input, (1, 1))
            action_HDC = HDC_model.predict([frame, vel_input])
            action_HDC = action_HDC[0]
            action_HDC_array.append(action_HDC)
            next_state, reward, done, _, _ = env.step(action_HDC)
            states_HDC.append(next_state[0].astype(np.float32))
            for o in range(steps - 1):
                #print("current states:", env.state)
                image = env.render()
                frame = process_image(image)
                velocity = env.state[1]
                # print("previous image shape:", frame.shape)
                # print("previous velocity shape:", velocity.shape)
                # change the input size
                # frame = np.expand_dims(frame, axis=-1)
                # frame = np.expand_dims(frame, axis=0)
                frame = np.reshape(frame, (1, 64, 64, 1))
                velocity = np.reshape(velocity, (1, 1))
                # print("Nowadays image shape:", frame.shape)
                # print("Nowadays velocity shape:", velocity.shape)

                predicted_actions = HDC_model.predict([frame, velocity])
                action_HDC_array.append(predicted_actions[0])
                next_state, reward, done, _, _ = env.step(predicted_actions)
                if isinstance(next_state[0], np.ndarray) and next_state[0].shape == (1,):
                    position = next_state[0]
                    position = position[0]
                    states_HDC.append(position)
                else:
                    states_HDC.append(next_state[0])

                # if next_state[0].size == 1:
                #     position = next_state[0]
                #     position = position[0]
                #     states_HDC.append(position)
                # else:
                #     states_HDC.append(next_state[0])
            one_trajectory_max_diff = []
            for q in range(len(states_LDC)):
                one_trajectory_max_diff.append(abs(states_LDC[q] - states_HDC[q]))

            one_action_max_diff = []
            for p in range(len(action_LDC_array)):
                one_action_max_diff.append(abs(action_LDC_array[p] - action_HDC_array[p]))

            #one_trajectory_max_diff = [abs(a - b) for a, b in zip(states_LDC, states_HDC)]
            sorted_traj = sorted(one_trajectory_max_diff)
            sorted_action = sorted(one_action_max_diff)
            max_diff_traj.append(max(one_trajectory_max_diff))
            max_diff_action.append(sorted_action[96])

        #CP value of the trajectory-based
        index = int(len(max_diff_traj) * 0.04)
        sorted_traj_data_CP = sorted(max_diff_traj)
        top_96_values = sorted_traj_data_CP[-index:]
        top_96_percent = top_96_values[0]
        CP_table[j][i] = top_96_percent
        #CP value for the action-based
        index = int(len(max_diff_action) * 0.04)
        sorted_action_CP_all = sorted(max_diff_action)
        action_96_CP = sorted_action_CP_all[-index]
        CP_action_table[j][i] = action_96_CP


        # index = int(len(diff_sort) * 0.04)
        # # Get the top 4% values, which is equivalent to the top 96% values
        # top_96_percent_values = diff_sort[-index:]
        # top_96_percent = top_96_percent_values[0]
        # CP_table[j][i] = top_96_percent


#save the table
with open('CP_table.txt', 'w') as f:
    for row in CP_table:
        f.write('\t'.join(map(str, row)) + '\n')

with open('New_action_CP_table.txt', 'w') as f:
    for row in CP_action_table:
        f.write(','.join(map(str, row)) + '\n')