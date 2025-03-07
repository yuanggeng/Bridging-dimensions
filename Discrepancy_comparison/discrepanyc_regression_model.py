import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

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

#test on my own dataset
dis_pos_start = -0.6; dis_pos_end = -0.5
dis_vel_start = -0.01; dis_vel_end = 0.02
points = 2000
file_dis_action_train = 'training_data/discrep_data/discrep_action_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start, dis_pos_end,
                                                                                                    dis_vel_start, dis_vel_end,
                                                                                                    points)
file_traject_train = 'training_data/discrep_data/discrep_trajectory_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start, dis_pos_end,
                                                                                                    dis_vel_start, dis_vel_end,
                                                                                                points)

file_vel_train = 'training_data/discrep_data/discrep_vel_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start, dis_pos_end,
                                                                                                    dis_vel_start, dis_vel_end,
                                                                                                points)
file_pos_train = 'training_data/discrep_data/discrep_pos_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start, dis_pos_end,
                                                                                                    dis_vel_start, dis_vel_end,
                                                                                                points)



test_points = 1000
file_dis_action_test = 'training_data/discrep_data/discrep_action_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start, dis_pos_end,
                                                                                                    dis_vel_start, dis_vel_end,
                                                                                                    test_points)
file_traject_test = 'training_data/discrep_data/discrep_trajectory_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start, dis_pos_end,
                                                                                                    dis_vel_start, dis_vel_end,
                                                                                                test_points)

file_vel_test = 'training_data/discrep_data/discrep_vel_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start, dis_pos_end,
                                                                                                    dis_vel_start, dis_vel_end,
                                                                                                test_points)
file_pos_test = 'training_data/discrep_data/discrep_pos_({0},{1})({2},{3})_{4}.npy'.format(dis_pos_start, dis_pos_end,
                                                                                                    dis_vel_start, dis_vel_end,
                                                                                                test_points)

act_disc_train  = np.load(file_dis_action_train)
pos_train = np.load(file_pos_train)
vel_train = np.load(file_vel_train)

act_disc_test  = np.load(file_dis_action_test)
pos_test = np.load(file_pos_test)
vel_test = np.load(file_vel_test)

train_x = np.vstack((pos_train, vel_train))
train_x = np.transpose(train_x)
test_x =  np.vstack((pos_test, vel_test))
test_x = np.transpose(test_x)


train_y = act_disc_train
test_y_real = act_disc_test



# act_disc = np.load(file_dis_action_train)
# random_positions = np.random.uniform(dis_pos_start, dis_pos_end, points)
# random_velocities = np.random.uniform(dis_vel_start, dis_vel_end, points)
#
# train_y = act_disc[0:200]
# test_y_real = act_disc[200:300]
#
# train_x = np.vstack((random_positions[0:200], random_velocities[0:200]))
# train_x = np.transpose(train_x)
# test_x = np.vstack((random_positions[200:300], random_velocities[200:300]))
# test_x = np.transpose(test_x)


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(train_x, train_y)




mu, cov = gpr.predict(test_x, return_cov=True)
test_y = mu.ravel()
uncertainty = 1.96 * np.sqrt(np.diag(cov))

mu_train, cov_train = gpr.predict(train_x, return_cov=True)
test_y_train = mu.ravel()
uncertainty_train = 1.96 * np.sqrt(np.diag(cov_train))

# test how many data within the confidence interval
sat_point = 0
for i in range(len(test_y)):
    if test_y_real[i] <= test_y[i]  + uncertainty[i]  and test_y_real[i] >= test_y[i]  - uncertainty[i] :
        sat_point += 1




plt.figure()
plt.title("l=%.2f sigma_f=%.2f" % (gpr.kernel_.k2.length_scale, gpr.kernel_.k1.constant_value))
plt.fill_between(test_x.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
plt.plot(test_x, test_y, label="predict")
plt.scatter(test_x, train_y, label="train", c="red", marker="x")
plt.legend()


#################stop here

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y):
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y).ravel()  # Ensure y is a 1D array
        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # Kernel of training data
        Kyy = self.kernel(X, X)  # Kernel of test data
        Kfy = self.kernel(self.train_X, X)  # Cross kernel of train and test data
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # Add a small noise for numerical stability

        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        # Sum over the input dimensions for each point, and expand dims to enable broadcasting
        sqdist = np.sum(x1 ** 2, axis=1).reshape(-1, 1) + np.sum(x2 ** 2, axis=1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * sqdist)






# Generate 2D test data
x = np.linspace(0, 5, 100)
y = np.linspace(0, 5, 100)
test_X = np.array([[xi, yi] for xi in x for yi in y])

gpr = GPR()
gpr.fit(train_X, train_y)
mu, cov = gpr.predict(test_X)

# For plotting, you might want to use contour plots since the data is 2D
X0, X1 = np.meshgrid(x, y)
Z = mu.reshape(X0.shape)

plt.figure()
plt.contourf(X0, X1, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, s=50, cmap='viridis')
plt.title("Gaussian Process Regression on 2D Data")
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()

# Generating synthetic data
X = np.random.uniform(-3., 3., (500, 1))
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Defining the kernel (RBF in this case)
kernel = RBF(length_scale=2.0)

# Instantiate a Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

# Fit to data
gp.fit(X, y)

# Make the prediction (including the standard deviation)
X_test = np.linspace(-3, 3, 1000).reshape(-1, 1)
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plot the function, the prediction, and the 95% confidence interval
plt.figure()
plt.plot(X_test, y_pred, 'k', lw=2, zorder=9)
plt.fill_between(X_test[:, 0], y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='k')
plt.scatter(X, y, c='r', s=50, zorder=10)
plt.title("Gaussian Process Regression")
plt.tight_layout()
plt.show()

a = 1
b = 1