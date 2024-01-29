import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import gym
import cv2
import time
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras import layers, Model
import gym
import cv2
from tensorflow import keras
import yaml
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

def process_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (64, 64))
    image = np.float32(np.true_divide(image, 255))
    return image


#given variable
HDC_model = tf.keras.models.load_model('trained_HDC_cnn_model.h5')
num_point = 100
pos_start = -0.6
pos_end = -0.4
vel_start = -0.02
vel_end = 0.07
def get_trajectory_training_data(HDC_model, pos_start, pos_end, vel_start, vel_end,num_point):
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
    #inside variable
    num_position = 2
    num_velocity = 2
    steps = 100
    positon_CP = np.linspace(pos_start, pos_end, num_position)
    velocity_CP = np.linspace(vel_start, vel_end, num_velocity)

    random_start_position = positon_CP[0]
    random_end_position = positon_CP[1]
    random_start_velocity = velocity_CP[0]
    random_end_velocity = velocity_CP[1]

    random_positions = np.random.uniform(random_start_position, random_end_position, num_point)
    random_velocities = np.random.uniform(random_start_velocity, random_end_velocity, num_point)
    sampled_states = np.stack((random_positions, random_velocities), axis=-1)

    action_set = []
    diff_set = []
    states_HDC = []

    for k in range(num_point):
        states_HDC = []
        # calculate the trajectory by LDC
        desired_state = sampled_states[k]

        # calculate the trajectory by HDC
        states_HDC = []
        action_HDC_array = []
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
    return states_HDC, action_HDC_array

#Gather the training data
# states_HDC, action_HDC_array = get_trajectory_training_data(HDC_model, pos_start, pos_end, vel_start, vel_end,num_point)
# states_HDC = states_HDC[:-1]
# np.save('LDC_inputs_(-0.6,-0.4)(-0.02,0.07)_100.npy', states_HDC)
# np.save('LDC_output_(-0.6,-0.4)(-0.02,0.07)_100.npy', action_HDC_array)

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


#train the model
X_train_np = np.load('LDC_inputs_(-0.6,-0.4)(-0.02,0.07)_100.npy')
Y_train_np = np.load('LDC_output_(-0.6,-0.4)(-0.02,0.07)_100.npy')

X_train = torch.tensor(X_train_np, dtype=torch.float32)
Y_train = torch.tensor(Y_train_np, dtype=torch.float32)
# Hyperparameters
learning_rate = 0.001
epochs = 5000
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
        criterion = nn.MSELoss()
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

dump_model_dict('LDC_new_(-0.6,-0.4)(-0.02,0.07).yml', model)

torch.save(model, 'LDC_new_(-0.6,-0.4)(-0.02,0.07).pth')



# def LDC_architecture():
#     model = keras.Sequential([
#         layers.Input(shape=(2,)),  # Input layer for position and velocity
#         layers.Dense(16, activation='sigmoid'),  # First hidden layer with 16 neurons and sigmoid activation
#         layers.Dense(16, activation='sigmoid'),  # Second hidden layer with 16 neurons and sigmoid activation
#         layers.Dense(1, activation='tanh')  # Output layer with 1 neuron and tanh activation
#     ])
#
#     # Compile the model (you can choose the optimizer and loss function according to your task)
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model
#
# x_train = np.load('LDC_inputs_(-0.6,-0.4)(-0.02,0.07)_100.npy')
# y_train = np.load('LDC_output_(-0.6,-0.4)(-0.02,0.07)_100.npy')
# epochs = 100  # For example, you can adjust based on your needs
# batch_size = 32  # Typical values are 32, 64, 128, etc.
# LDC_model = LDC_architecture()
# # Train the model
# history = LDC_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.15)
# LDC_model.save("LDC_(-0.6,-0.4)(-0.02,0.07)_100.h5")


#train several LDCs for all kinds of
