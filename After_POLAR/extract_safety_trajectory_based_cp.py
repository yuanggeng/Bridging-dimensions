import shutil
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import time
import scipy.io

# Directory path
def extract_reachable_point(folder_path):
    # folder_path = "/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/reach_verification_multiLDCs/reach_verification"
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

folder_path = "/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Inverted_pendulum_trajectoryCP/outputs_LDCs_noCP_rtrajector/outputs_multi_LDCs_traject/reach_verification"
#safe_theta, safe_thetadot = extract_reachable_point(folder_path)
#
# np.savetxt('IP_1LDC_safe_theta_verify.txt', safe_theta)
# np.savetxt('IP_1LDC_safe_thetadot_verify.txt', safe_thetadot)

# move all the safe_points plt file into another folder
folder_path_tube = "/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Inverted_pendulum_trajectoryCP/outputs_LDCs_noCP_rtrajector/outputs_multi_LDCs_traject/outputs"
destination = "/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Inverted_pendulum_trajectoryCP/extracted_safe_plt1"

def move_safe_plt(folder_path_tube, destination):
    #pattern = r"(\d+\.\d+)" #this is for the every number is above zero
    pattern = r'[-+]?\d+\.\d+'#this is for Single_Pendulum_1_1.080000_-1.880000.plt
    for i, element in enumerate(safe_theta):
        print(i, safe_theta[i], safe_thetadot[i])
        for plt_name in os.listdir(folder_path_tube):
            matches = re.findall(pattern, plt_name)
            # Extract the numbers from the desired positions
            one_theta = float(matches[0])
            one_dot = float(matches[1])
            #print(one_theta)  # Output: 1.05
            #print(one_dot)  # Output: 0.24

            if one_theta == safe_theta[i] and one_dot == safe_thetadot[i]:
                print("we are moving plt.")
                full_path_plt_name = os.path.join(folder_path_tube, plt_name)
                destination_path = os.path.join(destination, plt_name)
                shutil.move(full_path_plt_name, destination_path)

#move_safe_plt(folder_path_tube, destination)


# calculate the trajectory-based conformal prediction
time1 = time.time()

folder_path = "/home/UFAD/yuang.geng/yuang_research/Yuang_simulation/Yuang_simulation/Inverted_pendulum_trajectoryCP/extracted_safe_plt1"
#pattern = r"(\d+\.\d+)"
pattern = r'[-+]?\d+\.\d+'

min_array = []
max_array = []
traj_safe_theta = []
traj_safe_dot = []
failure_theta = []
failure_dot = []

for filename in os.listdir(folder_path):
    matches = re.findall(pattern, filename)
    # Extract the numbers from the desired positions
    one_theta = float(matches[0])
    one_dot = float(matches[1])
    # #load the conformal table
    # mat = scipy.io.loadmat('CI_theta-[5.5,6.2]_dot-[0,1]_700.mat')
    # conformal_table = mat['revised_matrix']
    # index_x = round((one_theta - 1) * 100)
    # index_y = round(one_dot * 100)
    specific_CI = 0.04
    # add conformal inference to the last reachable set
    file_location = os.path.join(folder_path, filename)
    with open(file_location, 'r') as file:
        data = file.read()
        lines = data.splitlines()
        last_set = lines[-12:-3]
        #second_column = [row[13:21] for row in last_set]
        second_column = [row[13:] for row in last_set]
        print(second_column)
        second_column_float = [float(x.strip()) for x in second_column]
        HDC_last_reach_high = [x + specific_CI for x in second_column_float]
        HDC_last_reach_low = [x - specific_CI for x in second_column_float]
        #print(HDC_last_reach)        # Test the safety after the conformal inference.
        max_value = max(HDC_last_reach_high)
        min_value = min(HDC_last_reach_low)
        min_array.append(min_value)
        max_array.append(max_value)
        if max_value <= 0.35 and min_value >= 0:
            traj_safe_theta.append(one_theta)
            traj_safe_dot.append(one_dot)
        else:
            failure_theta.append(one_theta)
            failure_dot.append(one_dot)



np.savetxt('IP_multiLDCs_trajectory_cp_safe_theta_verify.txt', traj_safe_theta)
np.savetxt('IP_multiLDCs_trajectory_cp_safe_thetadot_verify.txt', traj_safe_dot)
time2 = time.time()
gaptime = time2- time1
print("total time for reading the file is:", gaptime)