#add environment to PYTHONPATH
import sys
import os
env_path = os.path.join(os.path.abspath(os.getcwd()), '..\\Environments\\ContinuousCartPole')
sys.path.append(env_path)
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from continuous_cartpole import ContinuousCartPoleEnv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras
from tensorflow.keras.models import load_model
import time
import gym
from tqdm import tqdm_notebook
from collections import deque

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
# Load the trained model
class PolicyNetwork(nn.Module):

    # Takes in observations and outputs actions mu and sigma
    def __init__(self, observation_space):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, 2)

    # forward pass
    def forward(self, x):
        # input states
        x = self.input_layer(x)

        x = F.relu(x)

        # actions
        action_parameters = self.output_layer(x)

        return action_parameters

def select_action(network, state):
    ''' Selects an action given state
    Args:
    - network (Pytorch Model): neural network used in forward pass
    - state (Array): environment state

    Return:
    - action.item() (float): continuous action
    - log_action (float): log of probability density of action

    '''

    # create state tensor
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    state_tensor.required_grad = True

    # forward pass through network
    action_parameters = network(state_tensor)

    # get mean and std, get normal distribution
    mu, sigma = action_parameters[:, :1], torch.exp(action_parameters[:, 1:])
    m = Normal(mu[:, 0], sigma[:, 0])

    # sample action, get log probability
    action = m.sample()
    log_action = m.log_prob(action)

    return action.item(), log_action, mu[:, 0].item(), sigma[:, 0].item()

env = ContinuousCartPoleEnv()

# Gau_model = PolicyNetwork(env.observation_space.shape[0])
# Gau_model.load_state_dict(torch.load('model_RL_[-0.5,0.5].pth'))
# upper_bound = env.observation_space.high
# # Access the lower bound of the observation space
# lower_bound = env.observation_space.low
def get_train_1LDC(points):
    number_of_points = points
    rand_pos = np.random.uniform(low=-0.5, high=0.5, size=(number_of_points, 1))
    rand_theta = np.random.uniform(low=-0.2, high=0.2, size=(number_of_points, 1))
    rand_vel = np.random.uniform(low=-0.4, high=0.4, size=(number_of_points, 1))
    rand_dtheta = np.random.uniform(low=-0.3, high=0.3, size=(number_of_points, 1))
    input_states = [rand_pos, rand_vel, rand_theta, rand_dtheta]
    action_list = []
    for i in range(number_of_points):
        init_state = np.array([rand_pos[i], rand_vel[i], rand_theta[i], rand_dtheta[i]]).squeeze()
        action, la, mu, sigma = select_action(Gau_model, init_state)
        action = min(max(-1, action), 1)
        action = np.array([action], dtype=np.float32)
        action_list.append(action)

    np.save('training_X1.npy', input_states); np.save('training_Y1.npy', action_list)
    return input_states, action_list

# train_points = 15000
# input_states, action_list = get_train_1LDC(train_points)

class Control_NN(nn.Module):

    def __init__(self, layer_1_size=16, layer_2_size=16):
        super(Control_NN, self).__init__()
        self.fc1 = nn.Linear(4, layer_1_size)
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

def get_train_1LDC(points):
    number_of_points = points
    rand_pos = np.random.uniform(low=-0.5, high=0.5, size=(number_of_points, 1))
    rand_theta = np.random.uniform(low=-0.2, high=0.2, size=(number_of_points, 1))
    rand_vel = np.random.uniform(low=-0.4, high=0.4, size=(number_of_points, 1))
    rand_dtheta = np.random.uniform(low=-0.3, high=0.3, size=(number_of_points, 1))
    input_states = [rand_pos, rand_vel, rand_theta, rand_dtheta]
    action_list = []
    for i in range(number_of_points):
        init_state = np.array([rand_pos[i], rand_vel[i], rand_theta[i], rand_dtheta[i]]).squeeze()
        action, la, mu, sigma = select_action(Gau_model, init_state)
        action = min(max(-1, action), 1)
        action = np.array([action], dtype=np.float32)
        action_list.append(action)

    np.save('training_X1.npy', input_states); np.save('training_Y1.npy', action_list)
    return input_states, action_list

# train_points = 15000
# input_states, action_list = get_train_1LDC(train_points)
class Control_NN5(nn.Module):

    def __init__(self, layer_1_size=16, layer_2_size=16, layer_3_size=16, layer_4_size=16):
        super(Control_NN5, self).__init__()
        self.fc1 = nn.Linear(4, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc3 = nn.Linear(layer_2_size, layer_3_size)
        self.fc4 = nn.Linear(layer_3_size, layer_4_size)
        self.fc5 = nn.Linear(layer_4_size, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x
def dump_model_dict5(yml_filename, network: Control_NN5):
    model_dict = {}
    model_dict['activations'] = {}
    model_dict['activations'][1] = 'Sigmoid'
    model_dict['activations'][2] = 'Sigmoid'
    model_dict['activations'][3] = 'Sigmoid'
    model_dict['activations'][4] = 'Sigmoid'
    model_dict['activations'][5] = 'Tanh'
    model_dict['weights'] = {}
    model_dict['offsets'] = {}
    for layer in [1, 2, 3, 4, 5]:
        model_dict['weights'][layer] = network.state_dict()[f'fc{layer}.weight'].tolist()
        model_dict['offsets'][layer] = network.state_dict()[f'fc{layer}.bias'].tolist()
    with open(yml_filename, 'w') as f:
        yaml.dump(model_dict, f)
    return
########## start the main code####################3
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
        #tf.keras.layers.Dense(16, activation='sigmoid'),               # Fourth hidden layer
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



#----------test on LDC------------ check match with POLAR
#LDC_whole = load_LDC_model5('Cart_LDC1_whole_5layers.yml')
# LDC_whole = load_LDC_model('Cart_LDC1_whole_spce.yml')
#
# record_pos = []; record_rad = [];record_vel = []; record_angvel = []; record_action = []
# init_pos = 0; init_vel = 0; init_theta = 0.15; init_dtheta = -0.35 ; test_step = 20
# test_state = [init_pos, init_vel, init_theta, init_dtheta]
# init_state = np.array(test_state).squeeze()
# env.state = init_state
# init_state = np.reshape(init_state, (-1, 4))
#
# for _ in range(test_step):
#     current_state = np.reshape(env.state, (-1, 4))
#     action = LDC_whole.predict(current_state)
#     #action = np.array([0])
#     action = np.squeeze(action) # switch the action to (1,) format
#     action = np.array([action], dtype=np.float32)# switch the action to (1,) format
#     # update the state
#     state_variables, _, done, __ = env.step(action)
#     env.render()
#     record_action.append(action)
#     position = state_variables[0]; record_pos.append(position)
#     velocity = state_variables[1]; record_vel.append(velocity)
#     angle = state_variables[2]; record_rad.append(angle)
#     anglevelocity = state_variables[3]; record_angvel.append(anglevelocity)

#----------Get the ground truth of LDC-------------------

#----------Get the trajectory training data from LDC----------------




#----------train LDC-------------------

X_train1 = np.load('training_input_LDC_whole_12000.npy')
Y_train1 = np.load('training_output_LDC_whole_12000.npy')
# X_train1 = np.squeeze(X_train1).T

# X_train = np.concatenate((X_train1, X_train2), axis = 0)
# Y_train = np.concatenate((Y_train1, Y_train2), axis = 0)

#Plot the training dataset.
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot data
# x = X_train[:,0].reshape((25000, 1)); y = X_train[:, 2].reshape((25000, 1))
# z = Y_train
# ax.scatter(x, y, z)
# # Set labels
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# # Show plot
# plt.show()
X_train = torch.tensor(X_train1, dtype=torch.float32)
Y_train = torch.tensor(Y_train1, dtype=torch.float32)
# Hyperparameters
learning_rate = 0.0001
epochs = 300
batch_size = 6

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
        criterion = nn.MSELoss()
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

dump_model_dict('Cart_LDC1_3layers_new.yml', model)

torch.save(model, 'Cart_LDC1_3layers_new.pth')

# test on one LDC


#load the environment
start_time = time.time()
env = ContinuousCartPoleEnv()

Gau_model = PolicyNetwork(env.observation_space.shape[0])
Gau_model.load_state_dict(torch.load('model_RL_[-0.5,0.5].pth', map_location=torch.device('cpu')))
Gau_model.eval()

MAX_STEPS = 20
states_list = []
done = False
score = 0; test_points = 5000
safe_point = []
unsafe_point = []
for i in range(test_points):
    #this is by the random start
    # state = env.reset()
    # init_state = state
    # states_list.append(state)
    rand_pos = np.random.uniform(low=0.5, high=0.55, size=(1, 1))
    rand_theta = np.random.uniform(low=0.5, high=0.55, size=(1, 1))
    rand_vel = np.random.uniform(low=-0.4, high=0.4, size=(1, 1))
    rand_dtheta = np.random.uniform(low=-0.3, high=0.3, size=(1, 1))
    state = [rand_pos, rand_vel, rand_theta, rand_dtheta]
    state = np.array(state).squeeze()
    init_state = state
    env.state = state

    for step in range(MAX_STEPS):
            # env.render()
            action, la, mu, sigma = select_action(Gau_model, state)
            action = min(max(-1, action), 1)
            action = np.array([action], dtype=np.float32)

            new_state, reward, done, info = env.step(action)
            states_list.append(new_state)

            score += reward
            state = new_state
            # if done:
            #     break

    if state[2] >= -0.2 and state[2] <= 0.2:
        safe_point.append(init_state)
    else:
        unsafe_point.append(init_state)
env.close()
pos_safe = [arr[0] for arr in safe_point]
theta_safe = [arr[2] for arr in safe_point]

pos_unsafe = [arr[0] for arr in unsafe_point]
theta_unsafe = [arr[2] for arr in unsafe_point]

plt.scatter(pos_safe, theta_safe, color = 'green')
plt.scatter(pos_unsafe, theta_unsafe, color = 'red')

plt.title('test the safety')
plt.xlabel('time steps')
plt.ylabel('Position')
# Display the plot
plt.show()

np.save('safe_point_test.npy', safe_point)
np.save('unsafe_point_test.npy', unsafe_point)
end_time = time.time()
duration = end_time - start_time
print(f"Execution time: {duration} seconds")

#def get_train_LDC(points):

#def get_GT(init_set, steps, interval):


#keep running: -2.4 < x < 2.4, -0.418 < theta < 0.418
pos = [arr[0] for arr in states_list]
theta = [arr[2] for arr in states_list]

t = [i for i in range(len(pos))]

plt.plot(t, pos, label='Position')
plt.plot(t, theta, label='Theta')
plt.title('Position movement plot')
plt.xlabel('time steps')
plt.ylabel('Position')
# Display the plot
plt.show()
