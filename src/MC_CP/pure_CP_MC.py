import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import gym
import cv2
import yaml


text = []
text.append(1)
text.append(2)
np.savetxt('array.txt', text)


def process_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (64, 64))
    image = np.float32(np.true_divide(image, 255))
    return image

# load the model
env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env
HDC_model = tf.keras.models.load_model('trained_HDC_cnn_model.h5')
# define the interval
num_position = 11
# num_velocity = 10
#num_position = 6
num_velocity = 5
num_point = 60
threshold_CP = 58
steps = 100
positon_CP = np.linspace(-0.6, -0.4, num_position)
velocity_CP = np.linspace(-0.02, 0.06, num_velocity)
#velocity_CP.append(0.07)
velocity_CP = np.hstack((velocity_CP, 0.07))
split_pos = 5
split_vel = 11

safe_CP_points = []
safe_CP_position = []
safe_CP_velocity = []
# simulation
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
        check_robust = 0
        states_HDC = []
        for k in range(num_point):
            states_HDC = []
            desired_state = sampled_states[k]

            states_HDC = []
            action_HDC_array = []
            int_state = sampled_states[k]
            states_HDC.append(int_state[0])

            env.reset(specific_state=int_state)
            image = env.render()
            frame = process_image(image)
            vel_input = sampled_states[k][0]
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

                print(len(states_HDC))
                    #check the robustness

            if  states_HDC[-1] >= 0.45:
                check_robust += 1

        #if it's safe region
        if check_robust >= threshold_CP:
            this_loop_position = np.linspace(random_start_position, random_end_position, split_pos)
            split_vel = 1 + (random_end_velocity - random_start_velocity)/0.001
            split_vel = int(split_vel)
            this_loop_velocity = np.linspace(random_start_velocity, random_end_velocity, split_vel)
            for p in range(split_pos):
                for v in range(split_vel):
                    one_safe_point = [this_loop_position[p], this_loop_velocity[v]]
                    #safe_CP_points.append(one_safe_point)
                    safe_CP_position.append(round(this_loop_position[p], 4))
                    #safe_CP_velocity = safe_CP_points[:, 1]
                    safe_CP_velocity.append(round(this_loop_velocity[v], 4))


np.save('2PureCP_safe_position_(-0.6, -0.5)_(vel:-0.02-0.07).npy', safe_CP_position)
np.save('2PureCP_safe_velocity_(-0.6, -0.5)_(vel:-0.02-0.07).npy', safe_CP_velocity)

np.savetxt('2PureCP_safe_position.txt', safe_CP_position)
np.savetxt('2PureCP_safe_velocity.txt', safe_CP_velocity)

# safe_CP_position = safe_CP_points[:, 0]
# safe_CP_velocity = safe_CP_points[:, 1]



# check the robustness