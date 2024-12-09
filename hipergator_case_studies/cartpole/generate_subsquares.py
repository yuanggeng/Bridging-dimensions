import numpy as np

# Define the input range for the original 4D square
x_start, x_end = 0, 0.1
y_start, y_end = 0, 0.1
z_start, z_end = 0.06, 0.11
w_start, w_end = -0.4, -0.35

# Number of splits per dimension (e.g., 5x5x5x5 = 625 subspaces)
num_splits = 10

# Generate evenly spaced points with rounding to 3 decimal places
x_points = np.round(np.linspace(x_start, x_end, num_splits + 1), 3)
y_points = np.round(np.linspace(y_start, y_end, num_splits + 1), 3)
z_points = np.round(np.linspace(z_start, z_end, num_splits + 1), 3)
w_points = np.round(np.linspace(w_start, w_end, num_splits + 1), 3)

# Generate all the sub-squares
for i in range(num_splits):
    for j in range(num_splits):
        for k in range(num_splits):
            for l in range(num_splits):
                x1, x2 = x_points[i], x_points[i + 1]
                y1, y2 = y_points[j], y_points[j + 1]
                z1, z2 = z_points[k], z_points[k + 1]
                w1, w2 = w_points[l], w_points[l + 1]
                print(f"{x1} {x2} {y1} {y2} {z1} {z2} {w1} {w2}")
