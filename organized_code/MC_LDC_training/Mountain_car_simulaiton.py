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
model_LDC = load_LDC_model('sig16x16.yml')

def process_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (64, 64))
    image = np.float32(np.true_divide(image, 255))
    return image

def obtain_random_training_data(model, num_samples = 10000):
    #env = gym.make('MountainCarContinuous-v0')
    env = gym.make('MountainCarContinuous-v0', render_mode = 'rgb_array').env
    action_high = env.action_space.high
    action_low = env.action_space.low
    lower_bounds = env.observation_space.low
    upper_bounds = env.observation_space.high

    position_bounds = [lower_bounds[0], upper_bounds[0]]
    velocity_bounds = [lower_bounds[1], upper_bounds[1]]
    velocity_bounds = [0, 0.07]

    #print("position_bounds:", position_bounds)
    #print("velocity_bounds:", velocity_bounds)

    #num_samples = 10000
    # Randomly sample position and velocity
    random_positions = np.random.uniform(position_bounds[0], position_bounds[1], num_samples)
    random_velocities = np.random.uniform(velocity_bounds[0], velocity_bounds[1], num_samples)

    # Stack them together to get the sampled states
    sampled_states = np.stack((random_positions, random_velocities), axis=-1)

    image_set = []
    action_set = []

    for i in range(num_samples):
        desired_state = sampled_states[i]
        env.reset(specific_state=desired_state)
        state_array = desired_state.reshape(1, -1)
        #print(env.state)

        image = env.render()
        # plt.imshow(image)
        # plt.axis('off')  # check the correctness of image
        # plt.show()
        frame = process_image(image)
        image_set.append(frame)

        action = model.predict(state_array)
        action = action[0]
        action_set.append(action)

    return image_set, action_set, random_positions, random_velocities


# images_training, actions_training, positions_training, velocities_training = obtain_random_training_data(model)
# # np.save('image_training_data.npy', images_training)
# np.save('action_training_data.npy', actions_training)
# np.save('position_training_data.npy', positions_training)
# np.save('velocity_training_data.npy', velocities_training)

env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
lower_bounds = env.observation_space.low
upper_bounds = env.observation_space.high
position_bounds = [lower_bounds[0], upper_bounds[0]]
velocity_bounds = [lower_bounds[1], upper_bounds[1]]

start_time = time.time()
#def ground_truth_MC_HDC (model, )
HDC_model = tf.keras.models.load_model('trained_HDC_cnn_model.h5')
# velocity_values = np.linspace(-0.07, 0.06, int((0.07 - (-0.06)) / 0.01) + 1)
# position_values = np.linspace(-1.2, 0.59, int((0.6 - (-1.2)) / 0.01) + 1)
#velocity_values = np.linspace(-0.07, 0, int((0.07 - 0) / 0.01) + 1)
position_values = np.linspace(-0.6, -0.4, int((0.6 - 0.4) / 0.01) + 2)
velocity_values = np.linspace(0.05, 0.07, int((0.07 - 00.05) / 0.001) + 1)

num_pos = position_values.size
num_vel = velocity_values.size
print("the number position:", num_pos)
num_steps = 100
safe_states = []
unsafe_states = []
for i in range(num_pos):
    for j in range(num_vel):
        num_samples = 4
        now_pos = position_values[i]
        now_vel = velocity_values[j]
        x_samples = np.random.uniform(now_pos, now_pos + 0.01, num_samples)
        y_samples = np.random.uniform(now_vel, now_vel + 0.001, num_samples)

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
        print("finished initial:",i ,j)
            # x_axis = list(range(1, 31))
            # plt.figure()
            # plt.scatter(x_axis, y_steps)
            # plt.title('Position over Time')
            # plt.xlabel('Step')
            # plt.ylabel('Position')
            # plt.show()

np.save('2ground_truth_safe_states_(-0.6, -0.5)_(vel:-0.02-0.05).npy', safe_states)
np.save('2ground_truth_unsafe_states_(-0.6, -0.5)_(vel:-0.02-0.05).npy', unsafe_states)
end_time = time.time()
simu_time = end_time - start_time
print("total time for the simulation:", simu_time)





# env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
#
# safe_states = []
# unsafe_states = []
# image_set = []
# for i in range(1):
#
#     desired_state = np.array([-1, 0.06])
#     #desired_state = sampled_states[i]
#
#     env.reset(specific_state=desired_state)
#     #print(env.state)
#
#     image = env.render()
#     img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     # plt.imshow(image)
#     # plt.axis('off')  # Hide axes
#     # plt.show()
#
#     frame = process_image(image)
#     image_set.append(frame)
#     # plt.imshow(frame)
#     # plt.axis('off')  # Hide axes
#     # plt.show()
#
#     done = False
#
#     states = []
#     rewards = []
#     actions = []
#
#     states.append(desired_state)
#
#     for step in range(30):  # Run for 30 steps
#         # env.render()
#         if step == 0:
#             state_array = env.state.reshape(1, -1)
#         else:
#             state_array = state.reshape(1, -1)
#
#         action = model_LDC.predict(state_array)
#         action = action[0]
#
#         # Take a step in the environment
#         # next_state, reward, done = env.step(action)
#         next_state, reward, done, _, _ = env.step(action)
#
#         states.append(next_state)
#         rewards.append(reward)
#         actions.append(action)
#         # Update state
#         state = next_state
#
#
#     print(states[-1][0])
#     if states[-1][0] >= 0.45:
#         safe_states.append(desired_state)
#     else:
#         unsafe_states.append(desired_state)
#
#
#
#
# # Close the environment
# env.close()
#
# #plot the scatter
# x_safe, y_safe = zip(*safe_states)
# x_unsafe, y_unsafe = zip(*unsafe_states)
# plt.scatter(x_safe, y_safe, color='blue', marker='o', label='safe')
# plt.scatter(x_unsafe, y_unsafe, color='red', marker='o', label='unsafe')
#
# plt.title('simulation result')
# plt.xlabel('position')
# plt.ylabel('velocity')
# plt.legend()
# # Display the plot
# plt.show()
#
# states = np.array(states)
# rewards = np.array(rewards)
#
# # Example plot of position (state[0]) over time
# plt.figure()
# plt.plot(states[:, 0])
# plt.title('Position over Time')
# plt.xlabel('Step')
# plt.ylabel('Position')
# plt.show()
#
# # Example plot of velocity (state[1]) over time
# plt.figure()
# plt.plot(states[:, 1])
# plt.title('Velocity over Time')
# plt.xlabel('Step')
# plt.ylabel('Velocity')
# plt.show()
#
# # Example plot of reward over time
# plt.figure()
# plt.plot(rewards)
# plt.title('Reward over Time')
# plt.xlabel('Step')
# plt.ylabel('Reward')
# plt.show()
