import numpy as np
import matplotlib.pyplot as plt

# Load data from the .npy file
data = np.load('test.npy')  # Replace with your .npy file path

# Assuming the data is in the shape (num_samples, num_points, 2)
# Where the last dimension represents the x and y coordinates

num_samples = data.shape[0]

plt.figure(figsize=(10, 10))

# Plot each sample
for i in range(num_samples):
    plt.plot(data[i, :, 0], data[i, :, 1], label=f'Sample {i+1}')

plt.title('Plots from .npy File')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.show()
