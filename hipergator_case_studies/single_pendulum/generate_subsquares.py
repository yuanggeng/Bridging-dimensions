import numpy as np

# Define the input range for the original square
x_start, x_end = 0, 2
y_start, y_end = -2, 0

# Number of splits (10x10 grid = 100 sub-squares)
num_splits = 10

# Generate evenly spaced points with rounding to 1 decimal place
x_points = np.round(np.linspace(x_start, x_end, num_splits + 1), 1)
y_points = np.round(np.linspace(y_start, y_end, num_splits + 1), 1)

# Generate all the sub-squares
for i in range(num_splits):
    for j in range(num_splits):
        x1, x2 = x_points[i], x_points[i + 1]
        y1, y2 = y_points[j], y_points[j + 1]
        print(f"{x1} {x2} {y1} {y2}")

