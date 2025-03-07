# Gather the discrepancy data

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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

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

def get_dis_action_data(dis_pos_start, dis_pos_end, dis_vel_start, dis_vel_end, points, HDC, LDC):
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
    HDC_model = HDC
    LDC_model = LDC # load the yml file
    num_position = 2
    num_velocity = 2
    num_point = points
    steps = 60
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
                print(k)

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


    file_dis_action = 'switch_60steps_all/test_GPR_60/discrep_action_({0},{1})({2},{3})_{4}.npy'.format(
        round(dis_pos_start,3), round(dis_pos_end,3),
        round(dis_vel_start,3), round(dis_vel_end,3),
        points)
    file_traject = 'switch_60steps_all/test_GPR_60/discrep_trajectory_({0},{1})({2},{3})_{4}.npy'.format(
        round(dis_pos_start,3), round(dis_pos_end,3),
        round(dis_vel_start,3), round(dis_vel_end,3),
        points)

    file_pos = 'switch_60steps_all/test_GPR_60/discrep_pos_({0},{1})({2},{3})_{4}.npy'.format(
        round(dis_pos_start,3), round(dis_pos_end,3),
        round(dis_vel_start,3), round(dis_vel_end,3),
        points)
    file_vel = 'switch_60steps_all/test_GPR_60/discrep_vel_({0},{1})({2},{3})_{4}.npy'.format(
        round(dis_pos_start, 3), round(dis_pos_end, 3),
        round(dis_vel_start, 3), round(dis_vel_end, 3),
        points)
    np.save(file_pos, random_positions)
    np.save(file_vel, random_velocities)
    np.save(file_dis_action, max_diff_action)
    np.save(file_traject, max_diff_traj)
    return max_diff_traj, max_diff_action, random_positions, random_velocities

#Gather training data and test data for GPR
HDC = tf.keras.models.load_model('HDC_model/trained_HDC_cnn_model.h5')
train_points = 400
pos_start = [-0.6, -0.55]; vel_start = [-0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
# for i in range(6):
#     dis_pos_start1 = -0.6; dis_pos_end1 = -0.55; dis_vel_start1 = vel_start[i]; dis_vel_end1 = vel_start[i+1]
#     yml_name = 'switch_60steps_all/LDCs_60steps/LDC60_({0}, {1})-({2}, {3}).yml'.format(round(dis_pos_start1,3), round(dis_pos_end1, 3), round(dis_vel_start1,3), round(dis_vel_end1,3))
#     LDC = load_LDC_model(yml_name)
#     max_diff_traj1, max_diff_action1, random_positions1, random_velocities1 = get_dis_action_data(dis_pos_start1, dis_pos_end1, dis_vel_start1, dis_vel_end1, train_points, HDC, LDC)
#
# points = 60
# for i in range(6):
#     dis_pos_start1 = -0.6; dis_pos_end1 = -0.55; dis_vel_start1 = vel_start[i]; dis_vel_end1 = vel_start[i+1]
#     yml_name = 'switch_60steps_all/LDCs_60steps/LDC60_({0}, {1})-({2}, {3}).yml'.format(round(dis_pos_start1, 3), round(dis_pos_end1, 3), round(dis_vel_start1, 3), round(dis_vel_end1, 3))
#     LDC = load_LDC_model(yml_name)
#     max_diff_traj1, max_diff_action1, random_positions1, random_velocities1 = get_dis_action_data(dis_pos_start1, dis_pos_end1, dis_vel_start1, dis_vel_end1, points, HDC, LDC)
#


# test_pos = -0.6; test_vel = 0.023;test_points = 500
# file_pos_sub = 'training_data/discrep_data/test_for_GPR/discrep_pos_({0},{1})({2},{3})_{4}.npy'.format(
#     round(test_pos,3), round(test_pos + 0.01,3),
#     round(test_vel,3), round(test_vel + 0.001,3),
#     test_points)
# file_vel_sub = 'training_data/discrep_data/test_for_GPR/discrep_vel_({0},{1})({2},{3})_{4}.npy'.format(
#     round(test_pos,3), round(test_pos + 0.01,3),
#     round(test_vel,3), round(test_vel + 0.001,3),
#     test_points)
# test_pos4 = np.load(file_pos_sub); test_vel4 = np.load(file_vel_sub)
# test_pos = -0.6; test_vel1 = 0.024
# file_pos_sub = 'training_data/discrep_data/test_for_GPR/discrep_pos_({0},{1})({2},{3})_{4}.npy'.format(
#     round(test_pos,3), round(test_pos + 0.01,3),
#     round(test_vel1,3), round(test_vel1 + 0.001,3),
#     test_points)
# file_vel_sub = 'training_data/discrep_data/test_for_GPR/discrep_vel_({0},{1})({2},{3})_{4}.npy'.format(
#     round(test_pos,3), round(test_pos + 0.01,3),
#     round(test_vel1,3), round(test_vel1 + 0.001,3),
#     test_points)
# test_pos5 = np.load(file_pos_sub); test_vel5 = np.load(file_vel_sub)
# print(len(test_pos5))


# dis_pos_start = -0.6; dis_pos_end = -0.59
# dis_vel_start = 0.02; dis_vel_end = 0.021
# points = 500
# HDC = tf.keras.models.load_model('HDC_model/trained_HDC_cnn_model.h5')
# yml_name = 'LDC_models/multi_LDCs_training/test_GPR/LDC_({0}, {1})-({2}, {3}).yml'.format(round(dis_pos_start,3), round(dis_pos_end, 3), round(dis_vel_start,3), round(dis_vel_end,3))
# LDC = load_LDC_model(yml_name)
#
# #max_diff_traj, max_diff_action, random_positions, random_velocities = get_dis_action_data(dis_pos_start, dis_pos_end, dis_vel_start, dis_vel_end, points, HDC, LDC)
# dis_vel_start1 = 0.027; dis_vel_end1 = 0.028
# dis_vel_start2 = 0.028; dis_vel_end2 = 0.029
# dis_vel_start3 = 0.023; dis_vel_end3 = 0.024
# dis_vel_start4 = 0.024; dis_vel_end4 = 0.025
# dis_vel_start5 = 0.025; dis_vel_end5 = 0.026
# dis_vel_start6 = 0.026; dis_vel_end6 = 0.027
# dis_vel_start7 = 0.025; dis_vel_end7 = 0.026
# dis_vel_start8 = 0.026; dis_vel_end8 = 0.027
#
# # max_diff_traj1, max_diff_action1, random_positions1, random_velocities1 = get_dis_action_data(dis_pos_start, dis_pos_end, dis_vel_start7, dis_vel_end7, points, HDC, LDC)
# # max_diff_traj2, max_diff_action2, random_positions2, random_velocities2 = get_dis_action_data(dis_pos_start, dis_pos_end, dis_vel_start8, dis_vel_end8, points, HDC, LDC)
#
# #Gather the training or test data
# # max_diff_traj1, max_diff_action1, random_positions1, random_velocities1 = get_dis_action_data(dis_pos_start, dis_pos_end, dis_vel_start1, dis_vel_end1, points, HDC, LDC)
# # max_diff_traj2, max_diff_action2, random_positions2, random_velocities2 = get_dis_action_data(dis_pos_start, dis_pos_end, dis_vel_start2, dis_vel_end2, points, HDC, LDC)
# points2 = 400
# yml_name = 'LDC_models/multi_LDCs_training/test_GPR/LDC_({0}, {1})-({2}, {3}).yml'.format(round(dis_pos_start,3), round(dis_pos_end, 3), round(dis_vel_start3,3), round(dis_vel_end3,3))
# LDC = load_LDC_model(yml_name)
# #max_diff_traj3, max_diff_action3, random_positions3, random_velocities3 = get_dis_action_data(dis_pos_start, dis_pos_end, dis_vel_start3, dis_vel_end3, points2, HDC, LDC)
#
# yml_name = 'LDC_models/multi_LDCs_training/test_GPR/LDC_({0}, {1})-({2}, {3}).yml'.format(round(dis_pos_start,3), round(dis_pos_end, 3), round(dis_vel_start4,3), round(dis_vel_end4,3))
# LDC = load_LDC_model(yml_name)
# #max_diff_traj4, max_diff_action4, random_positions4, random_velocities4 = get_dis_action_data(dis_pos_start, dis_pos_end, dis_vel_start4, dis_vel_end4, points2, HDC, LDC)
# # max_diff_traj5, max_diff_action5, random_positions5, random_velocities5 = get_dis_action_data(dis_pos_start, dis_pos_end, dis_vel_start5, dis_vel_end5, points, HDC, LDC)
# # max_diff_traj6, max_diff_action6, random_positions6, random_velocities6 = get_dis_action_data(dis_pos_start, dis_pos_end, dis_vel_start6, dis_vel_end6, points, HDC, LDC)




#laod the training data

#train and test GPR
pos_start = -0.6; pos_end = -0.55
vel_start = -0.01; vel_end = 0.05
sub_points = 400
split_para = 7 #how many sub points within the interval?
sub_vel = np.linspace(vel_start, vel_end, split_para)
train_pos = np.array([]);train_vel= np.array([]);train_act = np.array([]); train_traj = np.array([])
for i in range(split_para-1):
    file_dis_action_sub = 'switch_60steps_all/test_GPR_60/discrep_action_({0},{1})({2},{3})_{4}.npy'.format(
        round(pos_start, 3), round(pos_end, 3),
        round(sub_vel[i], 3), round(sub_vel[i+1], 3),
        sub_points)
    file_traj_sub = 'switch_60steps_all/test_GPR_60/discrep_trajectory_({0},{1})({2},{3})_{4}.npy'.format(
        round(pos_start, 3), round(pos_end, 3),
        round(sub_vel[i], 3), round(sub_vel[i+1], 3),
        sub_points)

    file_pos_sub = 'switch_60steps_all/test_GPR_60/discrep_pos_({0},{1})({2},{3})_{4}.npy'.format(
        round(pos_start, 3), round(pos_end, 3),
        round(sub_vel[i], 3), round(sub_vel[i+1], 3),
        sub_points)
    file_vel_sub = 'switch_60steps_all/test_GPR_60/discrep_vel_({0},{1})({2},{3})_{4}.npy'.format(
        round(pos_start, 3), round(pos_end, 3),
        round(sub_vel[i], 3), round(sub_vel[i+1], 3),
        sub_points)
    train_pos1 = np.load(file_pos_sub); train_vel1 = np.load(file_vel_sub)
    train_act1 = np.load(file_dis_action_sub); train_traj1 = np.load(file_traj_sub)
    train_pos = np.append(train_pos, train_pos1); train_vel = np.append(train_vel, train_vel1)
    train_act = np.append(train_act, train_act1); train_traj = np.append(train_traj, train_traj1)

# aseced_act = np.sort(train_act)
# plt.hist(aseced_act[0:1000], bins=10, edgecolor='black', alpha=0.7)
# plt.hist(train_traj, bins=10, edgecolor='black', alpha=0.7)
#
# # Add titles and labels
# plt.title('Frequency Distribution')
# plt.xlabel('action discrepancy')
# plt.ylabel('Number of values')
# # Show the plot
# plt.show()


#train a GPR

train_x = np.vstack((train_pos, train_vel))
train_x = np.transpose(train_x)
train_y = train_act

kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(train_x, train_y)

def test_data_Gau(test_pos, test_vel,test_points, gpr):
    file_dis_action_sub = 'switch_60steps_all/test_GPR_60/test_60_data/discrep_action_({0},{1})({2},{3})_{4}.npy'.format(
        round(test_pos,3), round(test_pos + 0.05,3),
        round(test_vel,3), round(test_vel + 0.01,3),
        test_points)
    file_traj_sub = 'switch_60steps_all/test_GPR_60/test_60_data/discrep_trajectory_({0},{1})({2},{3})_{4}.npy'.format(
        round(test_pos,3), round(test_pos + 0.05,3),
        round(test_vel,3), round(test_vel + 0.01,3),
        test_points)

    file_pos_sub = 'switch_60steps_all/test_GPR_60/test_60_data/discrep_pos_({0},{1})({2},{3})_{4}.npy'.format(
        round(test_pos, 3), round(test_pos + 0.05, 3),
        round(test_vel, 3), round(test_vel + 0.01, 3),
        test_points)
    file_vel_sub = 'switch_60steps_all/test_GPR_60/test_60_data/discrep_vel_({0},{1})({2},{3})_{4}.npy'.format(
        round(test_pos, 3), round(test_pos + 0.05, 3),
        round(test_vel, 3), round(test_vel + 0.01, 3),
        test_points)
    test_pos1 = np.load(file_pos_sub); test_vel1 = np.load(file_vel_sub)
    test_act1 = np.load(file_dis_action_sub); test_traj1 = np.load(file_traj_sub)
    test_x = np.vstack((test_pos1, test_vel1))
    test_x = np.transpose(test_x)

    test_y_real = []
    mu, cov = gpr.predict(test_x, return_cov=True)
    test_y = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    # mu_train, cov_train = gpr.predict(train_x, return_cov=True)
    # test_y_train = mu.ravel()
    # uncertainty_train = 1.96 * np.sqrt(np.diag(cov_train))

    # test how many data within the confidence interval
    sat_point = 0
    for i in range(len(test_y)):
        if test_act1[i] <= test_y[i]  + uncertainty[i]  and test_act1[i] >= test_y[i]  - uncertainty[i] :
            sat_point += 1
    return sat_point
#set of the satisfiable points
sat_point1 = test_data_Gau(-0.6, -0.01, 150, gpr)
sat_point2 = test_data_Gau(-0.6, 0.0, 150, gpr)
sat_point3 = test_data_Gau(-0.6, 0.01, 150, gpr)
sat_point4 = test_data_Gau(-0.6, 0.02, 150, gpr)
sat_point5 = test_data_Gau(-0.6, 0.03, 150, gpr)
sat_point6 = test_data_Gau(-0.6, 0.04, 150, gpr)
print(sat_point1, sat_point2, sat_point3, sat_point4, sat_point5, sat_point6)
set_sat = []















file_dis_action = 'training_data/discrep_data/discrep_action_({0},{1})({2},{3})_{4}.npy'.format(
    dis_pos_start, dis_pos_end,
    dis_vel_start, dis_vel_end,
    points)
file_traject = 'training_data/discrep_data/discrep_trajectory_({0},{1})({2},{3})_{4}.npy'.format(
    dis_pos_start, dis_pos_end,
    dis_vel_start, dis_vel_end,
    points)

file_pos = 'training_data/discrep_data/discrep_pos_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start,
                                                                                      dis_pos_end,
                                                                                      dis_vel_start,
                                                                                      dis_vel_end,
                                                                                      points)
file_vel = 'training_data/discrep_data/discrep_vel_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start,
                                                                                      dis_pos_end,
                                                                                      dis_vel_start,
                                                                                      dis_vel_end,
                                                                                      points)
traj_disc_train = np.load(file_traject)
act_disc_train = np.load(file_dis_action)
pos_train = np.load(file_pos)
vel_train = np.load(file_vel)

ascend_order_action = np.sort(act_disc_train)
ascend_order_dis = np.sort(traj_disc_train)

#plt.hist(act_disc_train, bins=10, edgecolor='black', alpha=0.7)
plt.hist(traj_disc_train, bins=10, edgecolor='black', alpha=0.7)

# Add titles and labels
plt.title('Frequency Distribution')
plt.xlabel('action discrepancy')
plt.ylabel('Number of values')

# Show the plot
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.grid()
ax.scatter(traj_disc_train, vel_train, act_disc_train, c = 'r', s = 50)
ax.set_title('3D Scatter Plot')
# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
plt.show()