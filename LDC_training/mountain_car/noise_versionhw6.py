import numpy as np
import matplotlib.pyplot as plt
import collections
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

def process_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (64, 64))
    image = np.float32(np.true_divide(image, 255))
    return image

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
        sigma = 0.00001  # Standard deviation of the Gaussian noise
        # Assuming states_HDC is already populated with states
        noisy_states_HDC = []

    for state in states_HDC:
            noise = np.random.normal(0, sigma, state.shape)  # Generate Gaussian noise with the same shape as the state
            noisy_state = state + noise  # Add the noise to the state
            noisy_states_HDC.append(noisy_state)

    file_states = 'training_data/trajectory_training_data/LDC_inputs_({0},{1})({2},{3})_{4}.npy'.format(pos_start, pos_end, vel_start, vel_end, num_point)
    file_action = 'training_data/trajectory_training_data/LDC_output_({0},{1})({2},{3})_{4}.npy'.format(pos_start, pos_end, vel_start, vel_end, num_point)
    np.save(file_states, noisy_states_HDC)
    np.save(file_action, action_HDC_array)
    return noisy_states_HDC, action_HDC_array

def get_one_dis_test(dis_pos_start, dis_pos_end, dis_vel_start, dis_vel_end, points, HDC, LDC):
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
    HDC_model = HDC
    LDC_model = LDC # load the yml file
    num_position = 2
    num_velocity = 2
    num_point = points
    steps = 100
    positon_CP = np.linspace(dis_pos_start, dis_pos_end, num_position)
    velocity_CP = np.linspace(dis_vel_start, dis_vel_end, num_velocity)
    sigma = 0.00001

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
                    noise = np.random.normal(0, sigma, 2)
                    desired_state = next_state_LDC + noise

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

    return traj_CP, act_CP, sorted_traj_CP, sorted_action_CP_all

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


# pos_splited = [-0.6, -0.55, -0.5, -0.45, -0.4]
# vel_splited = [-0.02, 0, 0.02, 0.05]

pos_splited = [ -0.55, -0.54, -0.53, -0.52, -0.51 -0.5,]
vel_splited = [-0.02, 0,]

HDC = tf.keras.models.load_model('HDC_model/trained_HDC_cnn_model.h5')
num_point = 150; num_steps = 100
points = 60

traj_list = []; act_list = []
for i in range(len(pos_splited)-3):
    for j in range(len(vel_splited)-1):
        states_HDC, action_HDC_array = get_trajectory_training_data(HDC, pos_splited[i], pos_splited[i + 1], vel_splited[j],
                                                                    vel_splited[j + 1], num_point, num_steps)
        X_train_np = states_HDC
        Y_train_np = action_HDC_array
        LDC_smallest = train_LDC(X_train_np, Y_train_np)
        yml_name = 'LDC_models/multi_LDCs_training/noised_LDC/LDC_({0}, {1})-({2}, {3}).yml'.format(
            round(pos_splited[i], 3), round(pos_splited[i + 1], 3), round(vel_splited[j], 3),
            round(vel_splited[j + 1], 3))
        dump_model_dict(yml_name, LDC_smallest)
        torch_name = 'LDC_models/multi_LDCs_training/noised_LDC/LDC_({0}, {1})-({2}, {3}).pth'.format(
            round(pos_splited[i], 3), round(pos_splited[i + 1], 3), round(vel_splited[j], 3),
            round(vel_splited[j + 1], 3))
        torch.save(LDC_smallest, torch_name)

        LDC_dis = load_LDC_model(yml_name)
        traj_CP, act_CP, max_diff_traj, max_diff_action = get_one_dis_test(pos_splited[i], pos_splited[i + 1],vel_splited[j],
                                                                 vel_splited[j + 1], points, HDC, LDC_dis)
        act_list.append(act_CP)
        traj_list.append(traj_CP)


with open('LDC_models/multi_LDCs_training/noised_LDC/action_list.txt', 'w') as file:
    # Iterate over the list
    for item in act_list:
        # Write each item on a new line
        file.write(f"{item}\n")

with open('LDC_models/multi_LDCs_training/noised_LDC/trajectory_list.txt', 'w') as file:
    # Iterate over the list
    for item in traj_list:
        # Write each item on a new line
        file.write(f"{item}\n")

# # Parameters and distributions
# d, fq = 3, 0.3  # Alphabet size and geometric distribution parameter
# R = 1 / sum(fq**x for x in range(1, d + 1))  # Normalization factor
#
# def p(x, d):
#     return 1 / d
#
# def q(x, d, fq=fq, R=R):
#     return R * fq**x
#
# # Kullback-Leibler divergence functions
# def KL_divergence(func1, func2, d):
#     return sum(func1(x, d) * np.log(func1(x, d) / func2(x, d)) for x in range(1, d + 1))
#
# # Statistical functions
# def stat_LLR(pi_N, d):
#     return KL_divergence(p, q, d) + sum(pi_N[x-1] * np.log(q(x, d) / p(x, d)) for x in range(1, d + 1))
#
# def stat_H(pi_N, d):
#     return KL_divergence(p, q, d) + sum(pi_N[x-1] * np.log(pi_N[x-1] / p(x, d)) for x in range(1, d + 1))
#
# # Empirical distribution generator
# def pi_N_dist(d, N, distribution_type):
#     dist = [p(x, d) for x in range(1, d+1)] if distribution_type == "p" else [q(x, d) for x in range(1, d+1)]
#     x_data = np.random.choice(range(1, d+1), N, p=dist)
#     return [collections.Counter(x_data).get(x, 0) / N for x in range(1, d+1)]
#
# # Experiment setup
# M, W, N = 100, 100, 1000
#
# plt.bar(list(range(1, d+1)), pi_N_dist(d, N, "p"))
# plt.title("The distribution of p(x)")
# plt.show()
# plt.bar(list(range(1, d+1)), pi_N_dist(d, N, "q"))
# plt.title("The distribution of q(x)")
# plt.show()
#
#
# RE_pq, RE_qp = KL_divergence(p, q, d), KL_divergence(q, p, d)
# tau_list = np.linspace(0, RE_pq + RE_qp, num=M)
#
# # Lists for storing results
# results = {"LLR": {"P_p": [], "1-P_q": []}, "Hoeffding": {"P_p": [], "1-P_q": []}}
#
# # Main loop for simulations
# for tau in tau_list:
#     counters = {"LLR": {"P_p": 0, "1-P_q": 0}, "Hoeffding": {"P_p": 0, "1-P_q": 0}}
#     for _ in range(W):
#         for dist_type in ["p", "q"]:
#             pi_N = pi_N_dist(d, N, dist_type)
#             for test_type in ["LLR", "Hoeffding"]:
#                 stat = stat_LLR(pi_N, d) if test_type == "LLR" else stat_H(pi_N, d)
#                 if (stat >= tau and dist_type == "p") or (stat < tau and dist_type == "q"):
#                     counters[test_type]["P_p" if dist_type == "p" else "1-P_q"] += 1
#     for test_type in results:
#         results[test_type]["P_p"].append(counters[test_type]["P_p"] / W)
#         results[test_type]["1-P_q"].append(1 - counters[test_type]["1-P_q"] / W)
#
# # Plotting the results
# plt.plot(results["LLR"]["P_p"], results["LLR"]["1-P_q"], label="LLR Test")
# plt.plot(results["Hoeffding"]["P_p"], results["Hoeffding"]["1-P_q"], label="Hoeffding Test")
# plt.legend()
# plt.show()


