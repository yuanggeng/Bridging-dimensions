import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from keras.models import model_from_yaml
import yaml
import numpy as np
import tensorflow as tf
import time
import torch
import torch.nn as nn
import torch.optim as optim
# def load_LDC_model(filename='sig16x16.yml'):
#     # Load the neural network parameters from the YAML file
#     with open(filename, 'r') as file:
#         nn_params = yaml.load(file, Loader=yaml.FullLoader)
#
#     # Extract weights, biases, and activations
#     weights = nn_params['weights']
#     biases = nn_params['offsets']
#
#     # Convert lists to numpy arrays with correct shapes
#     for layer_index in weights.keys():
#         weights[layer_index] = np.array(weights[layer_index])
#         biases[layer_index] = np.array(biases[layer_index])
#
#     # Construct the model using TensorFlow and Keras
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(16, input_shape=(2,), activation='sigmoid'),  # Input layer
#         tf.keras.layers.Dense(16, activation='sigmoid'),                   # First hidden layer
#         tf.keras.layers.Dense(1, activation='tanh')                        # Output layer
#     ])
#
#     # Set the weights for each layer in the model
#     model.layers[0].set_weights([weights[1].transpose(), biases[1]])
#     model.layers[1].set_weights([weights[2].transpose(), biases[2]])
#     model.layers[2].set_weights([weights[3].transpose(), biases[3]])
#
#     # Compile the model with Adam optimizer and MSE loss
#     model.compile(optimizer='adam', loss='mean_squared_error')
#
#     return model
# model_LDC = load_LDC_model('sig16x16.yml')

def process_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (64, 64))
    image = np.float32(np.true_divide(image, 255))
    return image

#Get the ground truth of the MC
env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
lower_bounds = env.observation_space.low
upper_bounds = env.observation_space.high
position_bounds = [lower_bounds[0], upper_bounds[0]]
velocity_bounds = [lower_bounds[1], upper_bounds[1]]


# 60 steps
# pos_low = -0.6; pos_up = -0.5
# vel_low = -0.01; vel_up = 0.05

pos_low = -0.5; pos_up = -0.4
vel_low = -0.01; vel_up = 0.05

num_steps = 60
HDC_model = tf.keras.models.load_model('HDC_model/trained_HDC_cnn_model.h5')

def GT_MC_HDC(pos_low, pos_up, vel_low, vel_up, num_steps):
    start_time = time.time()
    # make the region seperated by interval 0.01 and 0.001.

    position_values = np.linspace(pos_low, pos_up, round((pos_up - pos_low) / 0.01) + 1)
    velocity_values = np.linspace(vel_low, vel_up, round((vel_up - vel_low) / 0.001) + 1)
    num_pos = position_values.size
    num_vel = velocity_values.size
    print("the number position:", num_pos)
    print("the number velocity:", num_vel)

    safe_states = []
    unsafe_states = []
    for i in range(num_pos - 1):

        for j in range(num_vel - 1):
            num_samples = 4
            now_pos = position_values[i]
            now_vel = velocity_values[j]
            now_pos_up = position_values[i+1]
            now_vel_up = velocity_values[j+1]
            x_samples = np.random.uniform(now_pos, now_pos_up, num_samples)
            y_samples = np.random.uniform(now_vel, now_vel_up, num_samples)

            end_states = []
            for k in range(num_samples):
                action_loop = []
                states_loop = []
                int_state = np.array([x_samples[k], y_samples[k]])
                states_loop.append(int_state)

                env.reset(specific_state=int_state)
                image = env.render()
                frame = process_image(image)
                vel_input = y_samples[k]
                frame = np.reshape(frame, (1, 64, 64, 1))
                vel_input = np.reshape(vel_input, (1, 1))
                action = HDC_model.predict([frame, vel_input])
                action = action[0]
                action_loop.append(action)

                next_state, reward, done, _, _ = env.step(action)
                states_loop.append(next_state)
                for o in range(num_steps - 1):
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
                    action_loop.append(predicted_actions[0])
                    next_state, reward, done, _, _ = env.step(predicted_actions)
                    states_loop.append(next_state)
                # print("the first element in the list:", states_loop[0][0])
                #y_simulation = states_loop[:][0]
                y_steps = [states_loop[i][0] for i in range(num_steps)]
                end_states.append(y_steps[-1])

                #check the safety
            result = all(value >= 0.45 for value in end_states)
            if result:
                safe_states.append([round(now_pos, 3), round(now_vel, 3)])
            else:
                unsafe_states.append([round(now_pos, 3), round(now_vel, 3)])

    filename_safe = 'switch_60steps_all/ground_truth/gt_60_safe_({0}, {1})_(vel:{2}-{3}).npy'.format(pos_low, pos_up, vel_low, vel_up)
    filename_unsafe = 'switch_60steps_all/ground_truth/gt_60_unsafe_({0}, {1})_(vel:{2}-{3}).npy'.format(pos_low, pos_up, vel_low, vel_up)
    np.save(filename_safe, safe_states)
    np.save(filename_unsafe, unsafe_states)
    end_time = time.time()
    simu_time = end_time - start_time
    print("total time for the simulation:", simu_time)
    return simu_time

#GT_time = GT_MC_HDC(pos_low, pos_up, vel_low, vel_up, num_steps)

# check what happened in these safety regions
# filename_safe = 'switch_60steps_all/ground_truth/gt_60_safe_({0}, {1})_(vel:{2}-{3}).npy'.format(pos_low, pos_up,
#                                                                                              vel_low, vel_up)
# filename_unsafe = 'switch_60steps_all/ground_truth/gt_60_unsafe_({0}, {1})_(vel:{2}-{3}).npy'.format(pos_low, pos_up,
#                                                                                                        vel_low, vel_up)
# safe_test1 = np.load(filename_safe)
# unsafe_test1 = np.load(filename_unsafe)
#
# safe_x = safe_test1[:, 0]
# safe_y = safe_test1[:, 1]
# unsafe_x = unsafe_test1[:, 0]
# unsafe_y = unsafe_test1[:, 1]
#
# plt.scatter(safe_x, safe_y, c='blue', label='Blue points')
# plt.scatter(unsafe_x, unsafe_y, c='red', label='Red points')
# plt.legend()
# # Optionally, add labels and title
# plt.xlabel('position')
# plt.ylabel('velocity')
# plt.title('safe and unsafe points')
# plt.show()

# combine the two subset together
# safe_test1 = np.load('training_data/ground_truth/ground_truth_safe_(-0.5, -0.4)_(vel:-0.01-0.06).npy')
# unsafe_test1 = np.load('training_data/ground_truth/ground_truth_unsafe_(-0.5, -0.4)_(vel:-0.01-0.06).npy')
# safe_test2 = np.load('training_data/ground_truth/ground_truth_safe_(-0.6, -0.5)_(vel:-0.01-0.06).npy')
# unsafe_test2 = np.load('training_data/ground_truth/ground_truth_unsafe_(-0.6, -0.5)_(vel:-0.01-0.06).npy')
#
# safe_all = np.concatenate((safe_test1, safe_test2))
# unsafe_all = np.concatenate((unsafe_test1, unsafe_test2))
#
# filename_safe_all = 'training_data/ground_truth/ground_truth_safe_(-0.4,-0.6)_(-0.01,0.06).npy'
# filename_unsafe_all = 'training_data/ground_truth/ground_truth_unsafe_(-0.4,-0.6)_(-0.01,0.06).npy'
# np.save(filename_safe_all, safe_all)
# np.save(filename_unsafe_all, unsafe_all)
#
# safe_x = safe_all[:, 0]
# safe_y = safe_all[:, 1]
# unsafe_x = unsafe_all[:, 0]
# unsafe_y = unsafe_all[:, 1]
#
# plt.scatter(safe_x, safe_y, c='blue', label='Blue points')
# plt.scatter(unsafe_x, unsafe_y, c='red', label='Red points')
# plt.legend()
# # Optionally, add labels and title
# plt.xlabel('position')
# plt.ylabel('velocity')
# plt.title('safe and unsafe points')
# plt.show()
#######################################

HDC_model = tf.keras.models.load_model('HDC_model/trained_HDC_cnn_model.h5')
# num_point = 4; num_steps = 50
# pos_start = -0.6; pos_end = -0.4
# vel_start = -0.01; vel_end = 0.06

num_point = 200; num_steps = 60
pos_start = -0.6; pos_end = -0.5
vel_start = -0.01; vel_end = 0.05


# Here we start the multiple LDCs training?
def get_trajectory_training_data(HDC_model, pos_start, pos_end, vel_start, vel_end, num_point, num_steps):
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
    num_position = 2
    num_velocity = 2
    steps = num_steps
    positon_CP = np.linspace(pos_start, pos_end, num_position)
    velocity_CP = np.linspace(vel_start, vel_end, num_velocity)

    random_start_position = positon_CP[0]
    random_end_position = positon_CP[1]
    random_start_velocity = velocity_CP[0]
    random_end_velocity = velocity_CP[1]

    random_positions = np.random.uniform(random_start_position, random_end_position, num_point)
    random_velocities = np.random.uniform(random_start_velocity, random_end_velocity, num_point)
    sampled_states = np.stack((random_positions, random_velocities), axis=-1)

    states_HDC = []
    action_HDC_array = []

    for k in range(num_point):
        desired_state = sampled_states[k]
        # calculate the trajectory by HDC

        int_state = sampled_states[k]
        states_HDC.append(int_state)

        #first state need to be specified
        env.reset(specific_state=int_state)
        image = env.render()
        frame = process_image(image)
        vel_input = sampled_states[k][1]  # the velocity should be the second one.
        frame = np.reshape(frame, (1, 64, 64, 1))
        vel_input = np.reshape(vel_input, (1, 1))
        action_HDC = HDC_model.predict([frame, vel_input])
        action_HDC = action_HDC[0]
        action_HDC_array.append(action_HDC)
        next_state, reward, done, _, _ = env.step(action_HDC)
        states_HDC.append(next_state.astype(np.float32))
        #states_HDC.append(next_state[0].astype(np.float32))
        for o in range(steps - 1):
            # print("current states:", env.state)
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
            if isinstance(next_state[0], np.ndarray) and next_state[0].shape == (1,) and isinstance(next_state[1], np.ndarray) and next_state[1].shape == (1,):
                position = next_state[0]
                position = position[0]
                velocity = next_state[1]
                velocity = velocity[0]
                this_state = np.stack((position, velocity), axis=-1)
                states_HDC.append(this_state)
                #states_HDC.append(next_state)
            else:
                states_HDC.append(next_state)

        pause = 0
        states_HDC = states_HDC[:-1]
        pause = 1

    file_states = 'training_data/trajectory_training_data/LDC_inputs_({0},{1})({2},{3})_{4}.npy'.format(pos_start, pos_end, vel_start, vel_end, num_point)
    file_action = 'training_data/trajectory_training_data/LDC_output_({0},{1})({2},{3})_{4}.npy'.format(pos_start, pos_end, vel_start, vel_end, num_point)
    np.save(file_states, states_HDC)
    np.save(file_action, action_HDC_array)
    return states_HDC, action_HDC_array

# Gather the training data
#states_HDC, action_HDC_array = get_trajectory_training_data(HDC_model, pos_start, pos_end, vel_start, vel_end, num_point, num_steps)

# states_HDC = states_HDC[:-1]

# file_states = 'training_data/trajectory_training_data/LDC_inputs_({0},{1})({2},{3})_{4}.npy'.format(pos_start, pos_end, vel_start, vel_end, num_point)
# file_action = 'training_data/trajectory_training_data/LDC_output_({0},{1})({2},{3})_{4}.npy'.format(pos_start, pos_end, vel_start, vel_end, num_point)
# training_output = np.load(file_action)
# training_inputs = np.load(file_states)
class Control_NN(nn.Module):

    def __init__(self, layer_1_size=16, layer_2_size=16):
        super(Control_NN, self).__init__()
        self.fc1 = nn.Linear(2, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc3 = nn.Linear(layer_2_size, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

def dump_model_dict(yml_filename, network: Control_NN):
    model_dict = {}
    model_dict['activations'] = {}
    model_dict['activations'][1] = 'Sigmoid'
    model_dict['activations'][2] = 'Sigmoid'
    model_dict['activations'][3] = 'Tanh'
    model_dict['weights'] = {}
    model_dict['offsets'] = {}
    for layer in [1, 2, 3]:
        model_dict['weights'][layer] = network.state_dict()[f'fc{layer}.weight'].tolist()
        model_dict['offsets'][layer] = network.state_dict()[f'fc{layer}.bias'].tolist()
    with open(yml_filename, 'w') as f:
        yaml.dump(model_dict, f)
    return

def train_LDC(X_train_np, Y_train_np):

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_np, dtype=torch.float32)
    # Hyperparameters
    learning_rate = 0.001
    epochs = 1000
    batch_size = 16

    # Initialize model, loss, and optimizer
    model = Control_NN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            # Get mini-batch
            inputs = X_train[i:i + batch_size]
            labels = Y_train[i:i + batch_size]
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    return model

# file_states = 'training_data/trajectory_training_data/LDC_inputs_({0},{1})({2},{3})_{4}.npy'.format(pos_start, pos_end, vel_start, vel_end, num_point)
# file_action = 'training_data/trajectory_training_data/LDC_output_({0},{1})({2},{3})_{4}.npy'.format(pos_start, pos_end, vel_start, vel_end, num_point)
# X_train_np = np.load(file_states)
# Y_train_np = np.load(file_action)
#model = train_LDC(X_train_np, Y_train_np)

# save the training model
# yml_name = 'LDC_models/trained_trajectory_0215/LDC_({0}, {1})-({2}, {3}).yml'.format(round(pos_start,2), round(pos_end,2), round(vel_start,2), round(vel_end,2))
# dump_model_dict(yml_name, model)
# torch_name = 'LDC_models/trained_trajectory_0215/LDC_({0}, {1})-({2}, {3}).pth'.format(round(pos_start,2), round(pos_end,2), round(vel_start,2), round(vel_end,2))
# torch.save(model, torch_name)

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

HDC_model =  tf.keras.models.load_model('HDC_model/trained_HDC_cnn_model.h5')

LDC_pos_lo = -0.6; LDC_pos_up = -0.5
LDC_vel_lo = -0.01; LDC_vel_up = 0.02
# yml_name = 'LDC_models/trained_trajectory_0215/LDC_({0}, {1})-({2}, {3}).yml'.format(round(LDC_pos_lo,2), round(LDC_pos_up, 2), round(LDC_vel_lo,2), round(LDC_vel_up,2))
# LDC_model = load_LDC_model(yml_name)
split_pos_num = 2; split_vel_num = 2
# def CP_traj_action(HDC_model, LDC_model,LDC_pos_lo,LDC_pos_up,LDC_vel_lo,LDC_vel_up,split_pos_num,split_vel_num):
#     env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
#     HDC_model = HDC_model
#     LDC_model = LDC_model # load the yml file
#     num_position = split_pos_num
#     num_velocity = split_vel_num
#     num_point = 60
#     steps = 50
#     # positon_CP = np.linspace(-0.6, -0.4, num_position)
#     # velocity_CP = np.linspace(-0.02, 0.07, num_velocity)
#     positon_CP = np.linspace(LDC_pos_lo, LDC_pos_up, num_position)
#     velocity_CP = np.linspace(LDC_vel_lo, LDC_vel_up, num_velocity)
#
#     CP_traj_table = [[0 for _ in range(num_position - 1)] for _ in range(num_velocity - 1)]
#     CP_action_table = [[0 for _ in range(num_position - 1)] for _ in range(num_velocity - 1)]
#
#     for i in range(num_position - 1):
#         for j in range(num_velocity - 1):
#             random_start_position = positon_CP[i]
#             random_end_position = positon_CP[i + 1]
#             random_start_velocity = velocity_CP[j]
#             random_end_velocity = velocity_CP[j + 1]
#
#             random_positions = np.random.uniform(random_start_position, random_end_position, num_point)
#             random_velocities = np.random.uniform(random_start_velocity, random_end_velocity, num_point)
#             sampled_states = np.stack((random_positions, random_velocities), axis=-1)
#
#             max_diff_traj = []
#             max_diff_action = []
#             for k in range(num_point):
#                 states_LDC = []
#                 action_LDC_array = []
#                 # calculate the trajectory by LDC
#                 states_LDC.append(sampled_states[k][0])
#                 desired_state = sampled_states[k]
#                 for m in range(steps):  # simulation for the LDC
#                     env.reset(specific_state=desired_state)
#                     state_array = desired_state.reshape(1, -1)
#
#                     action_LDC = LDC_model.predict(state_array)
#                     action_LDC = action_LDC[0]
#                     action_LDC_array.append(action_LDC)
#                     next_state_LDC, reward, done, _, _ = env.step(action_LDC)
#                     states_LDC.append(next_state_LDC[0])
#                     desired_state = next_state_LDC
#
#                 # calculate the trajectory by HDC
#                 states_HDC = []
#                 action_HDC_array = []
#                 int_state = sampled_states[k]
#                 states_HDC.append(int_state[0])
#
#                 env.reset(specific_state=int_state)
#                 image = env.render()
#                 frame = process_image(image)
#                 vel_input = sampled_states[k][1]  # the velocity should be the second one.
#                 frame = np.reshape(frame, (1, 64, 64, 1))
#                 vel_input = np.reshape(vel_input, (1, 1))
#                 action_HDC = HDC_model.predict([frame, vel_input])
#                 action_HDC = action_HDC[0]
#                 action_HDC_array.append(action_HDC)
#                 next_state, reward, done, _, _ = env.step(action_HDC)
#                 states_HDC.append(next_state[0].astype(np.float32))
#                 for o in range(steps - 1):
#                     # print("current states:", env.state)
#                     image = env.render()
#                     frame = process_image(image)
#                     velocity = env.state[1]
#                     # print("previous image shape:", frame.shape)
#                     # print("previous velocity shape:", velocity.shape)
#                     # change the input size
#                     # frame = np.expand_dims(frame, axis=-1)
#                     # frame = np.expand_dims(frame, axis=0)
#                     frame = np.reshape(frame, (1, 64, 64, 1))
#                     velocity = np.reshape(velocity, (1, 1))
#                     # print("Nowadays image shape:", frame.shape)
#                     # print("Nowadays velocity shape:", velocity.shape)
#
#                     predicted_actions = HDC_model.predict([frame, velocity])
#                     action_HDC_array.append(predicted_actions[0])
#                     next_state, reward, done, _, _ = env.step(predicted_actions)
#                     if isinstance(next_state[0], np.ndarray) and next_state[0].shape == (1,):
#                         position = next_state[0]
#                         position = position[0]
#                         states_HDC.append(position)
#                     else:
#                         states_HDC.append(next_state[0])
#
#
#                 one_trajectory_max_diff = []
#                 for q in range(len(states_LDC)):
#                     one_trajectory_max_diff.append(abs(states_LDC[q] - states_HDC[q]))
#
#                 one_action_max_diff = []
#                 for p in range(len(action_LDC_array)):
#                     one_action_max_diff.append(abs(action_LDC_array[p] - action_HDC_array[p]))
#
#                 sorted_traj = sorted(one_trajectory_max_diff)
#                 sorted_action = sorted(one_action_max_diff)
#                 max_diff_traj.append(max(one_trajectory_max_diff))
#                 max_diff_action.append(one_action_max_diff)
#                 # max_diff_action.append(sorted_action[96])
#
#             # CP value of the trajectory-based
#             index = int(len(max_diff_traj) * 0.04)
#             sorted_traj_data_CP = sorted(max_diff_traj)
#             top_96_values = sorted_traj_data_CP[-index:]
#             CP_traj_table[j][i] = top_96_values[0]
#             # CP value for the action-based
#             index = int(len(max_diff_action) * 0.04)
#             sorted_action_CP_all = sorted(max_diff_action)
#             action_96_CP = sorted_action_CP_all[-index]
#             CP_action_table[j][i] = action_96_CP
#
#
#     # save the table
#     with open('training_data/CP_data/CP_traj_table.txt', 'w') as f:
#         for row in CP_traj_table:
#             f.write(','.join(map(str, row)) + '\n')
#
#     with open('training_data/CP_data/CP_action_table.txt', 'w') as f:
#         for row in CP_action_table:
#             f.write(','.join(map(str, row)) + '\n')
#
#     return CP_traj_table, CP_action_table



# Obtain the random discrepancy over specific region
dis_pos_start = -0.6; dis_pos_end = -0.5
dis_vel_start = -0.01; dis_vel_end = 0.02
points = 2000
HDC = tf.keras.models.load_model('HDC_model/trained_HDC_cnn_model.h5')
# yml_name = 'LDC_models/trained_trajectory_0215/LDC_({0}, {1})-({2}, {3}).yml'.format(round(pos_start,2), round(pos_end, 2), round(vel_start,2), round(vel_end,2))
# LDC = load_LDC_model(yml_name)
def get_dis_action_data_old(dis_pos_start, dis_pos_end, dis_vel_start, dis_vel_end, points, HDC, LDC):
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
    HDC_model = HDC
    LDC_model = LDC # load the yml file
    num_position = 2
    num_velocity = 2
    num_point = points
    steps = 100
    positon_CP = np.linspace(dis_pos_start, dis_pos_end, num_position)
    velocity_CP = np.linspace(dis_vel_start, dis_vel_end, num_velocity)

    for i in range(num_position - 1):
        for j in range(num_velocity - 1):
            random_start_position = positon_CP[i]
            random_end_position = positon_CP[i + 1]
            random_start_velocity = velocity_CP[j]
            random_end_velocity = velocity_CP[j + 1]

            random_positions = np.random.uniform(random_start_position, random_end_position, num_point)
            random_velocities = np.random.uniform(random_start_velocity, random_end_velocity, num_point)
            sampled_states = np.stack((random_positions, random_velocities), axis=-1)

            max_diff_traj = []
            max_diff_action = []
            for k in range(num_point):
                states_LDC = []
                action_LDC_array = []
                # calculate the trajectory by LDC
                states_LDC.append(sampled_states[k][0])
                desired_state = sampled_states[k]
                for m in range(steps):  # simulation for the LDC
                    env.reset(specific_state=desired_state)
                    state_array = desired_state.reshape(1, -1)

                    action_LDC = LDC_model.predict(state_array)
                    action_LDC = action_LDC[0]
                    action_LDC_array.append(action_LDC)
                    next_state_LDC, reward, done, _, _ = env.step(action_LDC)
                    states_LDC.append(next_state_LDC[0])
                    desired_state = next_state_LDC

                # calculate the trajectory by HDC
                states_HDC = []
                action_HDC_array = []
                int_state = sampled_states[k]
                states_HDC.append(int_state[0])

                env.reset(specific_state=int_state)
                image = env.render()
                frame = process_image(image)
                vel_input = sampled_states[k][1]  # the velocity should be the second one.
                frame = np.reshape(frame, (1, 64, 64, 1))
                vel_input = np.reshape(vel_input, (1, 1))
                action_HDC = HDC_model.predict([frame, vel_input])
                action_HDC = action_HDC[0]
                action_HDC_array.append(action_HDC)
                next_state, reward, done, _, _ = env.step(action_HDC)
                states_HDC.append(next_state[0].astype(np.float32))
                for o in range(steps - 1):
                    # print("current states:", env.state)
                    image = env.render()
                    frame = process_image(image)
                    velocity = env.state[1]
                    frame = np.reshape(frame, (1, 64, 64, 1))
                    velocity = np.reshape(velocity, (1, 1))
                    predicted_actions = HDC_model.predict([frame, velocity])
                    action_HDC_array.append(predicted_actions[0])
                    next_state, reward, done, _, _ = env.step(predicted_actions)
                    if isinstance(next_state[0], np.ndarray) and next_state[0].shape == (1,):
                        position = next_state[0]
                        position = position[0]
                        states_HDC.append(position)
                    else:
                        states_HDC.append(next_state[0])

                one_trajectory_max_diff = []
                for q in range(len(states_LDC)):
                    one_trajectory_max_diff.append(abs(states_LDC[q] - states_HDC[q]))

                one_action_max_diff = []
                for p in range(len(action_LDC_array)):
                    one_action_max_diff.append(abs(action_LDC_array[p] - action_HDC_array[p]))

                max_diff_action.append(max(one_action_max_diff))
                max_diff_traj.append(max(one_trajectory_max_diff))
    return max_diff_traj, max_diff_action, random_positions, random_velocities

# max_diff_traj, max_diff_action, random_positions, random_velocities = get_dis_action_data(dis_pos_start, dis_pos_end, dis_vel_start, dis_vel_end, points, HDC, LDC)
#
# file_dis_action = 'training_data/discrep_data/discrep_action_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start, dis_pos_end,
#                                                                                                     dis_vel_start, dis_vel_end,
#                                                                                                     points)
# file_traject = 'training_data/discrep_data/discrep_trajectory_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start, dis_pos_end,
#                                                                                                     dis_vel_start, dis_vel_end,
#                                                                                                 points)
#
# file_pos = 'training_data/discrep_data/discrep_pos_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start, dis_pos_end,
#                                                                                                     dis_vel_start, dis_vel_end,
#                                                                                                     points)
# file_vel = 'training_data/discrep_data/discrep_vel_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start, dis_pos_end,
#                                                                                                     dis_vel_start, dis_vel_end,
#                                                                                                 points)
# np.save(file_pos, random_positions)
# np.save(file_vel, random_velocities)
# np.save(file_dis_action, max_diff_action)
# np.save(file_traject, max_diff_traj)


def get_one_dis_test(dis_pos_start, dis_pos_end, dis_vel_start, dis_vel_end, points, HDC, LDC,steps):
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
    HDC_model = HDC
    LDC_model = LDC # load the yml file
    num_position = 2
    num_velocity = 2
    num_point = points
    steps = steps
    positon_CP = np.linspace(dis_pos_start, dis_pos_end, num_position)
    velocity_CP = np.linspace(dis_vel_start, dis_vel_end, num_velocity)

    for i in range(num_position - 1):
        for j in range(num_velocity - 1):
            random_start_position = positon_CP[i]
            random_end_position = positon_CP[i + 1]
            random_start_velocity = velocity_CP[j]
            random_end_velocity = velocity_CP[j + 1]

            random_positions = np.random.uniform(random_start_position, random_end_position, num_point)
            random_velocities = np.random.uniform(random_start_velocity, random_end_velocity, num_point)
            sampled_states = np.stack((random_positions, random_velocities), axis=-1)

            max_diff_traj = []
            max_diff_action = []
            for k in range(num_point):
                states_LDC = []
                action_LDC_array = []
                # calculate the trajectory by LDC
                states_LDC.append(sampled_states[k][0])
                desired_state = sampled_states[k]
                for m in range(steps):  # simulation for the LDC
                    env.reset(specific_state=desired_state)
                    state_array = desired_state.reshape(1, -1)

                    action_LDC = LDC_model.predict(state_array)
                    action_LDC = action_LDC[0]
                    action_LDC_array.append(action_LDC)
                    next_state_LDC, reward, done, _, _ = env.step(action_LDC)
                    states_LDC.append(next_state_LDC[0])
                    desired_state = next_state_LDC

                # calculate the trajectory by HDC
                states_HDC = []
                action_HDC_array = []
                int_state = sampled_states[k]
                states_HDC.append(int_state[0])

                env.reset(specific_state=int_state)
                image = env.render()
                frame = process_image(image)
                vel_input = sampled_states[k][1]  # the velocity should be the second one.
                frame = np.reshape(frame, (1, 64, 64, 1))
                vel_input = np.reshape(vel_input, (1, 1))
                action_HDC = HDC_model.predict([frame, vel_input])
                action_HDC = action_HDC[0]
                action_HDC_array.append(action_HDC)
                next_state, reward, done, _, _ = env.step(action_HDC)
                states_HDC.append(next_state[0].astype(np.float32))
                for o in range(steps - 1):
                    # print("current states:", env.state)
                    image = env.render()
                    frame = process_image(image)
                    velocity = env.state[1]
                    frame = np.reshape(frame, (1, 64, 64, 1))
                    velocity = np.reshape(velocity, (1, 1))
                    predicted_actions = HDC_model.predict([frame, velocity])
                    action_HDC_array.append(predicted_actions[0])
                    next_state, reward, done, _, _ = env.step(predicted_actions)
                    if isinstance(next_state[0], np.ndarray) and next_state[0].shape == (1,):
                        position = next_state[0]
                        position = position[0]
                        states_HDC.append(position)
                    else:
                        states_HDC.append(next_state[0])

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


def get_dis_action_data(dis_pos_start, dis_pos_end, dis_vel_start, dis_vel_end, points, HDC, LDC, steps):
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
    HDC_model = HDC
    LDC_model = LDC # load the yml file
    num_position = 2
    num_velocity = 2
    num_point = points
    steps = steps
    positon_CP = np.linspace(dis_pos_start, dis_pos_end, num_position)
    velocity_CP = np.linspace(dis_vel_start, dis_vel_end, num_velocity)

    for i in range(num_position - 1):
        for j in range(num_velocity - 1):
            random_start_position = positon_CP[i]
            random_end_position = positon_CP[i + 1]
            random_start_velocity = velocity_CP[j]
            random_end_velocity = velocity_CP[j + 1]

            random_positions = np.random.uniform(random_start_position, random_end_position, num_point)
            random_velocities = np.random.uniform(random_start_velocity, random_end_velocity, num_point)
            sampled_states = np.stack((random_positions, random_velocities), axis=-1)

            max_diff_traj = []
            max_diff_action = []
            for k in range(num_point):
                states_LDC = []
                action_LDC_array = []
                # calculate the trajectory by LDC
                states_LDC.append(sampled_states[k][0])
                desired_state = sampled_states[k]
                for m in range(steps):  # simulation for the LDC
                    env.reset(specific_state=desired_state)
                    state_array = desired_state.reshape(1, -1)

                    action_LDC = LDC_model.predict(state_array)
                    action_LDC = action_LDC[0]
                    action_LDC_array.append(action_LDC)
                    next_state_LDC, reward, done, _, _ = env.step(action_LDC)
                    states_LDC.append(next_state_LDC[0])
                    desired_state = next_state_LDC

                # calculate the trajectory by HDC
                states_HDC = []
                action_HDC_array = []
                int_state = sampled_states[k]
                states_HDC.append(int_state[0])

                env.reset(specific_state=int_state)
                image = env.render()
                frame = process_image(image)
                vel_input = sampled_states[k][1]  # the velocity should be the second one.
                frame = np.reshape(frame, (1, 64, 64, 1))
                vel_input = np.reshape(vel_input, (1, 1))
                action_HDC = HDC_model.predict([frame, vel_input])
                action_HDC = action_HDC[0]
                action_HDC_array.append(action_HDC)
                next_state, reward, done, _, _ = env.step(action_HDC)
                states_HDC.append(next_state[0].astype(np.float32))
                for o in range(steps - 1):
                    # print("current states:", env.state)
                    image = env.render()
                    frame = process_image(image)
                    velocity = env.state[1]
                    frame = np.reshape(frame, (1, 64, 64, 1))
                    velocity = np.reshape(velocity, (1, 1))
                    predicted_actions = HDC_model.predict([frame, velocity])
                    action_HDC_array.append(predicted_actions[0])
                    next_state, reward, done, _, _ = env.step(predicted_actions)
                    if isinstance(next_state[0], np.ndarray) and next_state[0].shape == (1,):
                        position = next_state[0]
                        position = position[0]
                        states_HDC.append(position)
                    else:
                        states_HDC.append(next_state[0])

                one_trajectory_max_diff = []
                for q in range(len(states_LDC)):
                    one_trajectory_max_diff.append(abs(states_LDC[q] - states_HDC[q]))

                one_action_max_diff = []
                for p in range(len(action_LDC_array)):
                    one_action_max_diff.append(abs(action_LDC_array[p] - action_HDC_array[p]))

                max_diff_action.append(max(one_action_max_diff))
                max_diff_traj.append(max(one_trajectory_max_diff))


    file_dis_action = 'training_data/discrep_data/test_for_GPR/discrep_action_({0},{1})({2},{3})_{4}.npy'.format(
        round(dis_pos_start,3), round(dis_pos_end,3),
        round(dis_vel_start,3), round(dis_vel_end,3), points)
    file_traject = 'training_data/discrep_data/test_for_GPR/discrep_trajectory_({0},{1})({2},{3})_{4}.npy'.format(
        round(dis_pos_start,3), round(dis_pos_end,3),
        round(dis_vel_start,3), round(dis_vel_end,3), points)

    file_pos = 'training_data/discrep_data/test_for_GPR/discrep_pos_({0},{1})({2},{3})_{4}.npy'.format(
        round(dis_pos_start,3), round(dis_pos_end,3),
        round(dis_vel_start,3), round(dis_vel_end,3), points)
    file_vel = 'training_data/discrep_data/test_for_GPR/discrep_vel_({0},{1})({2},{3})_{4}.npy'.format(
        round(dis_pos_start,3), round(dis_pos_end,3),
        round(dis_vel_start,3), round(dis_vel_end,3), points)
    np.save(file_pos, random_positions)
    np.save(file_vel, random_velocities)
    np.save(file_dis_action, max_diff_action)
    np.save(file_traject, max_diff_traj)
    return max_diff_traj, max_diff_action, random_positions, random_velocities


# Multiple LDCs training
# alternative way for pure spltting for many LDCs
# pos_low_all = -0.5; pos_up_all = -0.4
# vel_low_all = -0.01; vel_up_all = 0.05
# num_steps = 60; num_point = 100 # usually 100 points for training. small can be 30
# HDC = tf.keras.models.load_model('HDC_model/trained_HDC_cnn_model.h5')
# pos_splited = np.linspace(pos_low_all, pos_up_all, 3)
# vel_splited = np.linspace(vel_low_all, vel_up_all, 7)
# points = 60 #200
# traj_list = []; act_list = []
#
# for i in range(3-1):
#     for j in range(7-1):
#         states_HDC, action_HDC_array = get_trajectory_training_data(HDC, pos_splited[i], pos_splited[i+1], vel_splited[j],
#                                                                     vel_splited[j+1], num_point, num_steps)
#         X_train_np = states_HDC; Y_train_np = action_HDC_array
#         LDC_smallest = train_LDC(X_train_np, Y_train_np)
#         yml_name = 'switch_60steps_all/LDCs_60steps/LDC60_({0}, {1})-({2}, {3}).yml'.format(
#             round(pos_splited[i], 3), round(pos_splited[i+1], 3), round(vel_splited[j], 3),
#             round(vel_splited[j+1], 3))
#         dump_model_dict(yml_name, LDC_smallest)
#         torch_name = 'switch_60steps_all/LDCs_60steps/LDC60_({0}, {1})-({2}, {3}).pth'.format(
#             round(pos_splited[i], 3), round(pos_splited[i+1], 3), round(vel_splited[j], 3),
#             round(vel_splited[j+1], 3))
#         torch.save(LDC_smallest, torch_name)
#
#         LDC_dis = load_LDC_model(yml_name)
#         cp_traj, cp_action, max_diff_traj, max_diff_action = get_one_dis_test(pos_splited[i],
#                         pos_splited[i+1], vel_splited[j],vel_splited[j+1], points, HDC, LDC_dis, num_steps)
#         act_list.append(cp_action)
#         traj_list.append(cp_traj)
#
#
# with open('switch_60steps_all/discrep_table/action_list.txt', 'w') as file:
#     # Iterate over the list
#     for item in act_list:
#         # Write each item on a new line
#         file.write(f"{item}\n")
#
# with open('switch_60steps_all/discrep_table/trajectory_list.txt', 'w') as file:
#     # Iterate over the list
#     for item in traj_list:
#         # Write each item on a new line
#         file.write(f"{item}\n")





# overall training process
# ONE LDC training
pos_low_all = -0.6; pos_up_all = -0.4
vel_low_all = -0.01; vel_up_all = 0.05
num_steps = 60; num_point = 600 # usually 100 points for training
HDC = tf.keras.models.load_model('HDC_model/trained_HDC_cnn_model.h5')

states_HDC, action_HDC_array = get_trajectory_training_data(HDC, pos_low_all, pos_up_all, vel_low_all, vel_up_all, num_point, num_steps)
X_train_np = states_HDC; Y_train_np = action_HDC_array
LDC_all = train_LDC(X_train_np, Y_train_np)
torch_name = 'switch_60steps_all/LDCs_60steps/LDC_({0}, {1})-({2}, {3}).pth'.format(round(pos_start,2), round(pos_end,2), round(vel_start,2), round(vel_end,2))
torch.save(LDC_all, torch_name)

points = 40 # at least 60 points for CP
#switch the LDC into the yml formal
yml_name = 'LDC_models/multi_LDCs_training/overall_yml/LDC_({0}, {1})-({2}, {3}).yml'.format(round(pos_low_all, 2), round(pos_up_all, 2), round(vel_low_all, 2), round(vel_up_all, 2))
dump_model_dict(yml_name, LDC_all)
LDC = load_LDC_model(yml_name)
traj_list = []; act_list = []

#defne which regeion you want for the mountain car.
traj_CP1, act_CP1, max_diff_traj1, max_diff_action1 = get_one_dis_test(-0.6, -0.4, 0.0, 0.01, points, HDC, LDC,num_steps)
act_list.append(act_CP1)
traj_list.append(traj_CP1)
traj_CP2, act_CP2, max_diff_traj2, max_diff_action2 = get_one_dis_test(-0.6, -0.4, 0.01, 0.02, points, HDC, LDC, num_steps)
act_list.append(act_CP2)
traj_list.append(traj_CP2)
traj_CP3, act_CP3, max_diff_traj3, max_diff_action3 = get_one_dis_test(-0.6, -0.4, 0.02, 0.03, points, HDC, LDC, num_steps)
act_list.append(act_CP3)
traj_list.append(traj_CP3)
traj_CP4, act_CP4, max_diff_traj4, max_diff_action4 = get_one_dis_test(-0.6, -0.4, 0.03, 0.04, points, HDC, LDC, num_steps)
act_list.append(act_CP4)
traj_list.append(traj_CP4)
traj_CP5, act_CP5, max_diff_traj5, max_diff_action5 = get_one_dis_test(-0.6, -0.4, 0.04, 0.05, points, HDC, LDC, num_steps)
act_list.append(act_CP5)
traj_list.append(traj_CP5)

#max_action_all = 0.1
thre_act = 0.05
act_CP =1;
def split_equally(pos_lo, pos_up, vel_lo, vel_up):
    pos_splited = np.linspace(pos_lo, pos_up, 3)
    vel_splited = np.linspace(vel_lo, vel_up, 3)
    pos_low_split = [pos_splited[0], pos_splited[1], pos_splited[0], pos_splited[1]]
    pos_up_split = [pos_splited[1], pos_splited[2], pos_splited[1], pos_splited[2]]
    vel_low_split = [vel_splited[0], vel_splited[0], vel_splited[1], vel_splited[1]]
    vel_up_split = [vel_splited[1], vel_splited[1], vel_splited[2], vel_splited[2]]

    return pos_low_split, pos_up_split, vel_low_split, vel_up_split

if act_CP < thre_act:
    yml_name = 'LDC_models/multi_LDCs_training/success_trained_model/LDC_({0}, {1})-({2}, {3}).yml'.format(round(pos_low_all, 2), round(pos_up_all, 2), round(vel_low_all, 2), round(vel_up_all, 2))
    dump_model_dict(yml_name, LDC)

if act_CP > thre_act:
    pos_lo_split, pos_up_split, vel_lo_split, vel_up_split = split_equally(pos_low_all, pos_up_all, vel_low_all, vel_up_all)

    for i in range(len(pos_lo_split)):
        states_HDC_sub, action_HDC_sub = get_trajectory_training_data(HDC, pos_lo_split[i], pos_up_split[i], vel_lo_split[i], vel_up_split[i], num_point, num_steps)
        X_train_np = states_HDC_sub; Y_train_np = action_HDC_sub
        LDC_sub_org = train_LDC(X_train_np, Y_train_np)

        #switch the LDC into the yml formal
        yml_name = 'LDC_models/multi_LDCs_training/overall_yml/LDC_({0}, {1})-({2}, {3}).yml'.format(round(pos_lo_split[i], 2), round(pos_up_split[i], 2), round(vel_lo_split[i], 2), round(vel_up_split[i], 2))
        dump_model_dict(yml_name, LDC_sub_org)
        LDC_sub = load_LDC_model(yml_name)
        traj_CP_sub, act_CP_sub, max_diff_traj, max_diff_action = get_one_dis_test(pos_lo_split[i], pos_up_split[i], vel_lo_split[i], vel_up_split[i], points, HDC, LDC_sub)
        if act_CP_sub <= thre_act:
            #we can save this model
            yml_name = 'LDC_models/multi_LDCs_training/success_trained_model/LDC_({0}, {1})-({2}, {3}).yml'.format(
                round(pos_lo_split[i], 2), round(pos_up_split[i], 2), round(vel_lo_split[i], 2),
                round(vel_up_split[i], 2))
            dump_model_dict(yml_name, LDC_sub_org)
            torch_name = 'LDC_models/multi_LDCs_training/success_trained_model/LDC_({0}, {1})-({2}, {3}).pth'.format(round(pos_lo_split[i], 2), round(pos_up_split[i], 2), round(vel_lo_split[i], 2),
                round(vel_up_split[i], 2))
            torch.save(LDC_sub_org, torch_name)

        if act_CP_sub > thre_act: #keep splitting
            pos_lo_split_2, pos_up_split_2, vel_lo_split_2, vel_up_split_2 = split_equally(pos_lo_split[i], pos_up_split[i], vel_lo_split[i], vel_up_split[i])
            for j in range(len(pos_lo_split_2)):
                states_HDC_sub2, action_HDC_sub2 = get_trajectory_training_data(HDC, pos_lo_split_2[j], pos_up_split_2[j], vel_lo_split_2[j], vel_up_split_2[j], num_point, num_steps)
                X_train2_np = states_HDC_sub2; Y_train2_np = action_HDC_sub2
                LDC_sub2 = train_LDC(X_train2_np, Y_train2_np)




                torch_name = 'LDC_models/multi_LDCs_training/success_trained_model/LDC_({0}, {1})-({2}, {3}).pth'.format(
                    round(pos_lo_split_2[j], 2), round(pos_up_split_2[j], 2), round(vel_lo_split_2[j], 2),
                    round(vel_up_split_2[j], 2))
                torch.save(LDC_sub2, torch_name)





                yml_name_sub2 = 'LDC_models/multi_LDCs_training/success_trained_model/LDC_({0}, {1})-({2}, {3}).yml'.format(
                    round(pos_lo_split_2[j], 3), round(pos_up_split_2[j], 3), round(vel_lo_split_2[j], 3),
                    round(vel_up_split_2[j], 3))
                dump_model_dict(yml_name_sub2, LDC_sub2)
                LDC_sub2 = load_LDC_model(yml_name)
                traj_CP_sub2, act_CP_sub2, max_diff_traj2, max_diff_action2 = get_one_dis_test(pos_lo_split_2[j],
                                                                                           pos_up_split_2[j],
                                                                                           vel_lo_split_2[j],
                                                                                           vel_up_split_2[j], points, HDC,
                                                                                           LDC_sub)
                #if act_CP_sub2 <= thre_act:
                # we can save this model
                # yml_name = 'LDC_models/multi_LDCs_training/success_trained_model/LDC_({0}, {1})-({2}, {3}).yml'.format(
                #     round(pos_lo_split_2[j], 2), round(pos_up_split_2[j], 2), round(vel_lo_split_2[j], 2),
                #     round(vel_up_split_2[j], 2))
                # dump_model_dict(yml_name, LDC_sub2)
                # torch_name = 'LDC_models/multi_LDCs_training/success_trained_model/LDC_({0}, {1})-({2}, {3}).pth'.format(
                #     round(pos_lo_split_2[j], 2), round(pos_up_split_2[j], 2), round(vel_lo_split_2[j], 2),
                #     round(vel_up_split_2[j], 2))
                # torch.save(LDC_sub2, torch_name)

# Neural network architecture search

