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
#
folder_path = "/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/organized_code/MC_after_POLAR/IP_POLAR_results/reach_verification_1LDC_traj_60/reach_verification"
extract_safe_theta, extract_safe_thetadot = extract_IP_reachable_point(folder_path)

states = np.array([extract_safe_theta, extract_safe_thetadot])

indices_above_1 = np.where(states[0, :] <= 1)

# Delete the entire columns at these indices
new_states = np.delete(states, indices_above_1, axis=1)

extract_safe_theta = new_states[0, :]
extract_safe_thetadot = new_states[1, :]

#np.savetxt('IP_POLAR_results/reach_verification_1LDC_traj_60/IP60_verified_position_1LDC.txt', extract_safe_theta)
#np.savetxt('IP_POLAR_results/reach_verification_1LDC_traj_60/IP60_verified_velocity_1LDC.txt', extract_safe_thetadot)


# Load the data
predicted_x = np.loadtxt('IP_POLAR_results/reach_verification_1LDC_traj_60/IP60_verified_position_1LDC.txt')
predicted_y = np.loadtxt('IP_POLAR_results/reach_verification_1LDC_traj_60/IP60_verified_velocity_1LDC.txt')
predicted_points = np.column_stack((predicted_x, predicted_y))


true_x = np.loadtxt('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/simulation_verification_results/IP_safe_theta_HDC.txt', delimiter=',')
true_y = np.loadtxt('/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/simulation_verification_results/IP_safe_dot_HDC.txt', delimiter=',')

# Step 2: Define a Matching Criterion (e.g., Euclidean distance)
all_true_dataset = np.column_stack((true_x, true_y))
total_population = (2/0.01) * (2/0.01)

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