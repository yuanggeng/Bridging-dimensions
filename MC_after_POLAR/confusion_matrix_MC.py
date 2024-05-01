import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import re


def extract_IP_reachable_point(folder_path):
    #folder_path = "/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/reach_verification_multiLDCs/reach_verification"
    # If the file name is "Yes_1.0500000.970000.txt"
    safe_theta = []
    safe_thetadot = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            # Extract the first 7 numbers
            first_number = float(filename.split("_")[1][:7])
            safe_theta.append(first_number)
            # Extract the last 7 numbers
            last_number = float(filename.split("_")[1][8:15])
            safe_thetadot.append(last_number)
    # np.savetxt('multi_safe_theta_verify.txt', safe_theta)
    # np.savetxt('multi_safe_thetadot_verify.txt', safe_thetadot)

    return safe_theta, safe_thetadot

# folder_path = "/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/organized_code/MC_after_POLAR/IP_POLAR_results/outputs_rebuttal_1-2/outputs/reach_verification"
# extract_safe_theta, extract_safe_thetadot = extract_IP_reachable_point(folder_path)
# np.savetxt('IP_POLAR_results/IP_noise_verified_position_new.txt', extract_safe_theta)
# np.savetxt('IP_POLAR_results/IP_noise_verified_velocity_new.txt', extract_safe_thetadot)
#
#
def extract_MC_reachable_point(folder_path):
    safe_theta = []
    safe_thetadot = []

    for filename in os.listdir(folder_path):
        print(f"Checking filename: {filename}")  # Debug print
        if filename.endswith(".txt"):
            matches = re.findall(r"(-?\d+\.\d+)", filename)
            print(f"Matches found: {matches}")  # Debug print

            if len(matches) == 2:
                theta, thetadot = map(float, matches)
                safe_theta.append(theta)
                safe_thetadot.append(thetadot)
            else:
                first_number = float(filename.split("_")[1][:7])
                safe_theta.append(first_number)
                # Extract the last 7 numbers
                last_number = float(filename.split("_")[1][8:15])
                safe_thetadot.append(last_number)
                print("len!=2",first_number, last_number)

    return safe_theta, safe_thetadot

def extract_values(folder_path):
    safe_theta = []
    safe_thetadot = []
    # Search for float values in the filename
    for filename in os.listdir(folder_path):
        matches = re.findall(r"(-?\d+\.\d+)", filename)

        if len(matches) == 2:
            value1, value2 = map(float, matches)
            safe_theta.append(value1)
            safe_thetadot.append(value2)
            return safe_theta, safe_thetadot
        else:
            return None, None

# folder_path = "MC_POLAR_results/reach_verification_traj1/reach_verification"
# extract_safe_theta, extract_safe_thetadot = extract_MC_reachable_point(folder_path)
# folder_path2 = "MC_POLAR_results/reach_verification_traj2/reach_verification"
# extract_safe_theta2, extract_safe_thetadot2 = extract_MC_reachable_point(folder_path2)
# extract_safe_theta = np.concatenate((extract_safe_theta, extract_safe_theta2), axis = 0)
# extract_safe_thetadot = np.concatenate((extract_safe_thetadot, extract_safe_thetadot2), axis = 0)
# np.savetxt('MC_POLAR_results/MC60_LDCs_traj_safe_position_verify.txt', extract_safe_theta)
# np.savetxt('MC_POLAR_results/MC60_LDCs_traj_safe_velocity_verify.txt', extract_safe_thetadot)

#extract 1LDC action_based verified safe points
# folder_path_1LDC_AB = "MC_POLAR_results/reach_verification_60AB_-0.5-0.4/reach_verification"
# extract_safe_theta, extract_safe_thetadot = extract_MC_reachable_point(folder_path_1LDC_AB)
# np.savetxt('MC_POLAR_results/MC60_LDCs_AB_safe_position_verify.txt', extract_safe_theta)
# np.savetxt('MC_POLAR_results/MC60_LDCs_AB_safe_velocity_verify.txt', extract_safe_thetadot)

# plt.scatter(extract_safe_theta, extract_safe_thetadot, color='blue', marker='o', label='Sample Data')
# # Add title and labels
# plt.title('Scatter Plot for verification')
# plt.xlabel('theta')
# plt.ylabel('tehta dot')
# plt.legend()
# plt.show()
#



# Function for drawing the plot of the verification
def draw_rec_plot(veri_theta_path, veri_dot_path):
    # verified_theta_x = np.loadtxt('simulation_verification_results/multi_safe_theta_verify.txt')
    # predicted_dot_y = np.loadtxt('simulation_verification_results/multi_safe_thetadot_verify.txt')
    verified_theta_x = np.loadtxt(veri_theta_path)
    predicted_dot_y = np.loadtxt(veri_dot_path)
    number_points = predicted_dot_y.shape[0]

    rectangles = np.random.rand(number_points, 4)
    rectangles[:, 0] = verified_theta_x
    rectangles[:, 1] = predicted_dot_y
    rectangles[:, 2] = 0.01
    rectangles[:, 3] = 0.01
    rectangles = np.round(rectangles, 2)
    fig, ax = plt.subplots()
    # Each element in rectangles is (x, y, width, height)
    for rect in rectangles:
        ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2], rect[3])) # 0,1 buttom left corrdinate. 2,3 are the width and height
    #Mountain car range
    ax.set_xlabel('velocity')
    ax.set_ylabel('position')
    ax.set_xlim(-0.07, 0.07)
    ax.set_ylim(-1.2, 0.0)
    #Inverted pendulum range
    plt.show()

theta_filepath = '/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/verificaiton_results/1LDC_v00.07_p64_cp0.0249_100steps/MC_1LDC_cp0.0249_safe_position_verify.txt'
thetadot_filepath = '/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/verificaiton_results/1LDC_v00.07_p64_cp0.0249_100steps/MC_1LDC_cp0.0249_safe_velocity_verify.txt'
#draw_rec_plot(theta_filepath, thetadot_filepath)




# Make a function for the confusion matrix
# Step 1: Load the Data
# #This is the one_LDC action-based conformal prediction with 60 steps
predicted_x = np.loadtxt('MC_POLAR_results/MC60_LDCs_AB_safe_position_verify.txt')
predicted_y = np.loadtxt('MC_POLAR_results/MC60_LDCs_AB_safe_velocity_verify.txt')

predicted_x = predicted_x[1:80]
predicted_y = predicted_y[1:80]

predicted_points = np.column_stack((predicted_x, predicted_y))

# #This is the one_LDC trajectory-based conformal prediction
# predicted_x = np.loadtxt('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/verificaiton_results/For_confusion_matrix/MC_1LDC_trajectory_cp_safe_theta_verify.txt')
# predicted_y = np.loadtxt('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/verificaiton_results/For_confusion_matrix/MC_1LDC_trajectory_cp_safe_thetadot_verify.txt')

#this is the Multiple LDCs action-based CP for new AB method
# predicted_x = np.loadtxt('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/reach_verified_AB/MC_LDCs_new_AB_safe_theta_verify.txt')
# predicted_y = np.loadtxt('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/reach_verified_AB/MC_LDCs_new_AB_safe_thetadot_verify.txt')
# predicted_points = np.column_stack((predicted_x, predicted_y))

#this is the multiple LDCs with trajectory-based
# predicted_x = np.loadtxt('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/MC_Multi_trajectory_cp_safe_pos_verify.txt')
# predicted_y = np.loadtxt('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/MC_Multi_trajectory_cp_safe_velocity_verify.txt')
# predicted_points = np.column_stack((predicted_x, predicted_y))
#
# raw_all_predicted_points = np.vstack((predicted_x, predicted_y))
# filter_small_range = raw_all_predicted_points[:, raw_all_predicted_points[1, :] <= 0.05]
# predicted_x = filter_small_range[0, :]
# predicted_y = filter_small_range[1, :]
#
# predicted_points = np.column_stack((predicted_x, predicted_y))
#
# max_velocity = max(predicted_y)
# min_velocity = min(predicted_y)
# max_position = max(predicted_x)
# min_position = min(predicted_x)


# 60 steps MC with 10 LDCs with the action-based discrepancy [-0.6, -0.4]x[-0.01, 0.05]
# predicted_x1 = np.loadtxt('MC_POLAR_results/MC60_LDCs_safe_position_verify.txt')
# predicted_y1 = np.loadtxt('MC_POLAR_results/MC60_LDCs_safe_velocity_verify.txt')
# predicted_x2 = np.loadtxt('MC_POLAR_results/MC60_LDCs_safe_position_verify_part2.txt')
# predicted_y2 = np.loadtxt('MC_POLAR_results/MC60_LDCs_safe_velocity_verify_part2.txt')
# predicted_x = np.concatenate((predicted_x1, predicted_x2), axis = 0)
# predicted_y = np.concatenate((predicted_y1, predicted_y2), axis = 0)
# predicted_points = np.column_stack((predicted_x, predicted_y))


# 60 steps LDC with trajectory-based discrepancy
# predicted_x = np.loadtxt('MC_POLAR_results/MC60_LDCs_traj_safe_position_verify.txt')
# predicted_y = np.loadtxt('MC_POLAR_results/MC60_LDCs_traj_safe_velocity_verify.txt')
#
# predicted_points = np.column_stack((predicted_x, predicted_y))
#
# raw_all_predicted_points = np.vstack((predicted_x, predicted_y))
# filter_small_range = raw_all_predicted_points[:, raw_all_predicted_points[1, :] <= 0.05]
# predicted_x = filter_small_range[0, :]
# predicted_y = filter_small_range[1, :]
#
# predicted_points = np.column_stack((predicted_x, predicted_y))
#
# max_velocity = max(predicted_y)
# min_velocity = min(predicted_y)
# max_position = max(predicted_x)
# min_position = min(predicted_x)


#Pure CP
# predicted_x = np.loadtxt('PureCP_safe_position.txt')
# predicted_y = np.loadtxt('PureCP_safe_velocity.txt')

# predicted_x = np.loadtxt('2PureCP_safe_position.txt')
# predicted_y = np.loadtxt('2PureCP_safe_velocity.txt')
#
# max_velocity = max(predicted_y)
# min_velocity = min(predicted_y)
# max_position = max(predicted_x)
# min_position = min(predicted_x)
#
# predicted_points = np.column_stack((predicted_x, predicted_y))
#
# #delete the value with 3 decimal
# mask = (predicted_points[:, 0] * 100).astype(int) == predicted_points[:, 0] * 100
# new_pair =predicted_points[mask]
# predicted_points = predicted_points[mask]
#
# #check the repetivity of prediction data
# unique_rows, indices, counts = np.unique(predicted_points, axis=0, return_index=True, return_counts=True)
# repetitive_elements = unique_rows[counts > 1]
# repetitive_indices = indices[counts > 1]
# mask = np.ones(predicted_points.shape[0], dtype=bool)
# mask[repetitive_indices] = False
# # Use the mask to filter the original array
# filtered_all_predicted_data = predicted_points[mask]
#
#
# predicted_points = filtered_all_predicted_data
# predicted_x = predicted_points[:,0]
# predicted_y = predicted_points[:,1]


#wrong with predicted values


#true_dataset1 = np.load('Mountain_car/ground_truth_safe_states_(vel:-0.07-0).npy')
#rue_dataset2 = np.load('Mountain_car/ground_truth_safe_states_(vel:0-0.07).npy')
#all_true_dataset = np.vstack((true_dataset1, true_dataset2))
# true_x = all_true_dataset[:, 0]
# true_y = all_true_dataset[:, 1]

# true_dataset1 = np.load('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/new_100step_groundtruth/ground_truth_safe_states_(-0.5, -0.4)_(vel:0-0.07).npy')
# true_dataset2 = np.load('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/new_100step_groundtruth/ground_truth_safe_states_(-0.6, -0.5)_(vel:0-0.07).npy')
# all_true_dataset = np.vstack((true_dataset1, true_dataset2))

# true_dataset1 = np.load('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/new_100step_groundtruth/2ground_truth_safe_states_(-0.6, -0.5)_(vel:-0.02-0.05).npy')
# true_dataset2 = np.load('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/new_100step_groundtruth/2ground_truth_safe_states_(-0.5, -0.4)_(vel:-0.02-0.05).npy')
# true_dataset3 = np.load('new_100step_groundtruth/2ground_truth_safe_states_(-0.6, -0.4)_(vel:-0.05-0.07).npy')
# all_true_dataset = np.vstack((true_dataset1, true_dataset2,true_dataset3))
# total_population = (0.2/0.01) * (0.09/0.001)

#here is the mountain car's ground truth [-0.6, -0.4]x[-0.02, 0.05]
# true_dataset1 = np.load('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/new_100step_groundtruth/2ground_truth_safe_states_(-0.6, -0.5)_(vel:-0.02-0.05).npy')
# true_dataset2 = np.load('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Mountain_car/new_100step_groundtruth/2ground_truth_safe_states_(-0.5, -0.4)_(vel:-0.02-0.05).npy')
# all_true_dataset = np.vstack((true_dataset1, true_dataset2))
# total_population = (0.2/0.01) * (0.07/0.001)

#here is the mountain car's ground truth 60 steps [-0.6, -0.5]x[-0.01, 0.05]
# true_dataset1_1 = np.load('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/organized_code/MC_LDC_training/switch_60steps_all/ground_truth/gt_60_safe_(-0.6, -0.5)_(vel:-0.01-0.05).npy')
# true_dataset2_1 = np.load('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/organized_code/MC_LDC_training/switch_60steps_all/ground_truth/gt_60_unsafe_(-0.6, -0.5)_(vel:-0.01-0.05).npy')
# true_dataset1_2 = np.load('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/organized_code/MC_LDC_training/switch_60steps_all/ground_truth/gt_60_safe_(-0.5, -0.4)_(vel:-0.01-0.05).npy')
# true_dataset2_2 = np.load('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/organized_code/MC_LDC_training/switch_60steps_all/ground_truth/gt_60_unsafe_(-0.5, -0.4)_(vel:-0.01-0.05).npy')
# true_safe_pos = np.concatenate((true_dataset1_1, true_dataset1_2), axis=0)
# true_safe_vel = np.concatenate((true_dataset2_1, true_dataset2_2), axis=0)

# np.save('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/organized_code/MC_LDC_training/switch_60steps_all/ground_truth/gt_60_safe_(-0.6, -0.4)_(vel:-0.01-0.05).npy',true_safe_pos)
# np.save('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/organized_code/MC_LDC_training/switch_60steps_all/ground_truth/gt_60_unsafe_(-0.6, -0.4)_(vel:-0.01-0.05).npy',true_safe_vel)


true_safe = np.load('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/organized_code/MC_LDC_training/switch_60steps_all/ground_truth/gt_60_safe_(-0.6, -0.4)_(vel:-0.01-0.05).npy')
true_unsafe= np.load('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/organized_code/MC_LDC_training/switch_60steps_all/ground_truth/gt_60_unsafe_(-0.6, -0.4)_(vel:-0.01-0.05).npy')
all_true_dataset = np.vstack((true_safe, true_unsafe))
total_population = (0.2/0.01) * (0.06/0.001)


#true_data set: check the repetitive rows and delete in the array

unique_rows, indices, counts = np.unique(all_true_dataset, axis=0, return_index=True, return_counts=True)
repetitive_elements = unique_rows[counts > 1]
repetitive_indices = indices[counts > 1]
mask = np.ones(all_true_dataset.shape[0], dtype=bool)
mask[repetitive_indices] = False

# Use the mask to filter the original array
filtered_all_true_data = all_true_dataset[mask]

true_x = filtered_all_true_data[:, 0]
true_y = filtered_all_true_data[:, 1]

#check all the predicted points are within the right region


# Step 2: Define a Matching Criterion (e.g., Euclidean distance)
# predicted_points = np.column_stack((predicted_x, predicted_y))
true_points = np.column_stack((true_x, true_y))

# check they are matched or not
plt.scatter(true_points[:, 0], true_points[:, 1], color='blue', marker='o', label='simulaiton_true')
plt.scatter(predicted_points[:, 0], predicted_points[:, 1], color='red', label='verification_predict')
plt.title('Scatter Plot for simulation')
plt.xlabel('theta')
plt.ylabel('tehta dot')
plt.legend()
plt.show()


# Step 3: Count True Positives and False Negatives
true_positives = 0
false_negatives = 0
threshold = 1e-6 #0  # Set an appropriate threshold
true_positive_x = []
true_positive_y = []
for tx, ty in zip(true_x, true_y):
    distances = np.sqrt((predicted_x - tx)**2 + (predicted_y - ty)**2)

    if np.min(distances) <= threshold:  # If there's a match within the threshold
        true_positives += 1
        true_positive_x.append(tx)
        true_positive_y.append(ty)
    else:
        false_negatives += 1

plt.scatter(true_positive_x, true_positive_y, color='red', label='TP points')
plt.title('Scatter Plot for true positive points')
plt.xlabel('position')
plt.ylabel('velocity')
plt.legend()
plt.show()

# for tx, ty in zip(true_x, true_y):
#     for m in range(predicted_x.size()):
#
#     distances = np.sqrt((predicted_x - tx)**2 + (predicted_y - ty)**2)
#     if np.min(distances) <= threshold:  # If there's a match within the threshold
#         true_positives += 1
#     else:
#         false_negatives += 1

# Step 4: Calculate TPR
true_positives = true_positives
tpr = true_positives / (true_positives + false_negatives)
print("True Positive Rate:", tpr)
all_predict_points = predicted_points.shape[0]
false_positives = all_predict_points - true_positives


# MC example
#total_population = (0.2/0.01) * (0.07/0.001) #this is previous state space
#total_population = (0.6 - 0.4)/0.01 * 0.07/0.001 # this is for initial state space (-0.6, -0.4),(0, 0.07)

#total_population = (0.2/0.01) * (0.09/0.001)

true_negative = total_population - false_positives



# Step5: draw confusion table
import numpy as np
import matplotlib.pyplot as plt

# Given values
TP = true_positives
TN = true_negative
FP = false_positives
FN = false_negatives

precision = true_positives/ (true_positives + false_positives)
print(f"Precision: {precision:.4f}")
FPR = false_positives / (false_positives + true_negative)
print(f"False Positive Rate: {FPR:.4f}")
recall = true_positives / (true_positives + false_negatives)
print(f"Recall: {recall:.4f}")
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"accuracy: {accuracy:.4f}")
f1_score = 2 * (precision * recall) / (precision + recall)
print("F1 Score:", f1_score)


# Constructing confusion matrix
conf_matrix = np.array([[TN, FP], [FN, TP]])

fig, ax = plt.subplots()

# Set the color map to 'coolwarm'
cax = ax.matshow(conf_matrix, cmap='coolwarm')

# Show color bar
fig.colorbar(cax)

# Set up labels for the two classes
class_labels = ['Negative', 'Positive']

# Set up axes
ax.set_xticklabels([''] + class_labels)
ax.set_yticklabels([''] + class_labels)
ax.xaxis.set_ticks_position('bottom')

# Annotate cells with the numerical values
for i in range(2):
    for j in range(2):
        ax.text(j, i, conf_matrix[i, j], ha='center', va='center')

# Labels and title
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Show the plot
plt.show()

