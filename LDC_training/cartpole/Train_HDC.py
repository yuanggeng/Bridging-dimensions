import time
import tensorflow as tf
from tensorflow.keras import layers, Model
import gym
import cv2
from tensorflow import keras
import yaml
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf


def process_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (64, 64))
    image = np.float32(np.true_divide(image, 255))
    return image
def test_simulation():
    step = 130
    model = tf.keras.models.load_model('trained_HDC_cnn_model.h5')
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env

    # given initial state
    states_HDC = []
    desired_state = np.array([-0.4, -0.07])
    states_HDC.append(desired_state.reshape(2, 1))
    env.reset(specific_state=desired_state)
    for i in range(step):
        print("current states:", env.state)
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

        predicted_actions = model.predict([frame, velocity])
        next_state, reward, done, _, _ = env.step(predicted_actions)
        states_HDC.append(next_state)
    return states_HDC

# states_HDC = test_simulation()
# first_one = states_HDC[0][0]
# second_one = states_HDC[1][0]
# allarray = states_HDC[0:2][0]
# positions = []
# for i in range(130):
#     positions.append(states_HDC[i][0])
# plt.figure()
# #first_elements = [arr[0, 0] for arr in states_HDC]
# x = np.arange(1, 131)
# plt.scatter(x, positions, color='blue', marker='o', label='Simulation result')
#
# #plt.plot(positions)
# plt.title('Position over 30 steps')
# plt.xlabel('Step')
# plt.ylabel('Position')
# plt.show()

# Load your trained model


# Extract layers from the model

def save_model_to_yaml(model_path, yaml_path):
    """
    Loads a model from an h5 file and saves its weights, biases, and activations to a yaml file.

    Args:
    - model_path (str): Path to the h5 model file.
    - yaml_path (str): Desired output path for the yaml file.
    """
    # Load the model
    model = load_model(model_path)

    # Extract weights, biases, and activations
    nn_params = {}
    nn_params['weights'] = []
    nn_params['biases'] = []
    nn_params['activations'] = []

    for layer in model.layers:
        weights = layer.get_weights()
        if weights:  # This checks if the layer has weights (like Dense, Conv2D layers, etc.)
            nn_params['weights'].append(weights[0].tolist())
            nn_params['biases'].append(weights[1].tolist())
            nn_params['activations'].append(layer.get_config()['activation'])

    # Save to YAML
    with open(yaml_path, 'w') as file:
        yaml.dump(nn_params, file)

#save_model_to_yaml("LDC1_whole_space.h5", "LDC1_whole_space.yml")

def HDC_model():
    image_input = layers.Input(shape=(64, 64, 1), name="image_input") # assuming grayscale image
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
    training_in_image = np.load('image_training_data.npy')
    training_out_action = np.load('action_training_data.npy')
    training_in_velocity = np.load('velocity_training_data.npy')

    # Train the model
    history = HDC_architect.fit(
        [training_in_image, training_in_velocity],  # Input data
        training_out_action,                        # Output data
        epochs=10,                                  # Number of epochs; adjust based on your needs
        batch_size=32,                              # Batch size; adjust based on your needs
        validation_split=0.2                        # Use 20% of the data for validation; adjust based on your needs
    )
    return HDC_architect

# Optionally save the model for future use
# HDC_architect.save('trained_HDC_cnn_model.h5')




# generate training dataset for LDC
def generate_trainingdata_one_LDC(model):
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
    lower_bounds = env.observation_space.low
    upper_bounds = env.observation_space.high
    #position_bounds = [lower_bounds[0], upper_bounds[0]]
    #velocity_bounds = [lower_bounds[1], upper_bounds[1]]
    position_bounds = [-0.6, 0.6]
    velocity_bounds = [0, 0.07]
    num_samples = 10000
    # Randomly sample position and velocity
    random_positions = np.random.uniform(position_bounds[0], position_bounds[1], num_samples)
    random_velocities = np.random.uniform(velocity_bounds[0], velocity_bounds[1], num_samples)
    # Stack them together to get the sampled states
    sampled_states = np.stack((random_positions, random_velocities), axis=-1)

    action_set = []
    for i in range(num_samples):
        desired_state = sampled_states[i]
        env.reset(specific_state=desired_state)
        state_array = desired_state.reshape(1, -1)

        image = env.render()
        frame = process_image(image)
        velocity = np.reshape(random_velocities[i], (1, 1))
        frame = np.reshape(frame, (1, 64, 64, 1))

        action = model.predict([frame, velocity])
        action = action[0]
        action_set.append(action)
    return sampled_states, action_set


time1 = time.time()
HDC_test = tf.keras.models.load_model('trained_HDC_cnn_model.h5')
input_states, action_output = generate_trainingdata_one_LDC(HDC_test)
np.save('3rdLDC_inputs_(0,0.07)(-0.6,0.6)_10000.npy', input_states)
np.save('3rdLDC_output_(0,0.07)(-0.6,0.6)_action_10000.npy', action_output)
time2 = time.time()
time_generate = time2 - time1
print('total time for generating 10000 samples:', time_generate)
#
#
# #Test the model in the simulation
# HDC_test = tf.keras.models.load_model('trained_HDC_cnn_model.h5')

def LDC_architecture():
    model = keras.Sequential([
        layers.Input(shape=(2,)),  # Input layer for position and velocity
        layers.Dense(16, activation='sigmoid'),  # First hidden layer with 16 neurons and sigmoid activation
        layers.Dense(16, activation='sigmoid'),  # Second hidden layer with 16 neurons and sigmoid activation
        layers.Dense(1, activation='tanh')  # Output layer with 1 neuron and tanh activation
    ])

    # Compile the model (you can choose the optimizer and loss function according to your task)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# x_train = np.load('1LDC_inputs_states_10000whole.npy')
# y_train = np.load('1LDC_output_action_10000whole.npy')
x_train = np.load('2ndLDC_inputs_(0,0.07)_10000.npy')
y_train = np.load('2ndLDC_output_(0,0.07)_action_10000.npy')
epochs = 100  # For example, you can adjust based on your needs
batch_size = 32  # Typical values are 32, 64, 128, etc.
LDC_model = LDC_architecture()
# Train the model
history = LDC_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.15)
LDC_model.save("LDC2nd_(0,0.07)_space.h5")








