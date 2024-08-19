# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load data from the .npy file
# data = np.load('test4.npy')  # Replace with your .npy file path
#
# # Assuming the data is in the shape (num_samples, num_points, 2)
# # Where the last dimension represents the x and y coordinates
# print(data.shape)
# num_samples = data.shape[0]
#
# plt.figure(figsize=(10, 10))
#
# # Plot each sample
# for i in range(num_samples):
#     plt.plot(data[i, :, 0], data[i, :, 1], label=f'Sample {i+1}')
#
# plt.title('Plots from .npy File')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.legend()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load data from the .npy file
# data = np.load('test4.npy')  # Replace with your .npy file path
#
# # Assuming the data is in the shape (num_samples, num_points, 2)
# # Where the last dimension represents the x and y coordinates
# print(data.shape)
# num_samples = data.shape[0]
#
# plt.figure(figsize=(10, 10))
#
# # Plot every 10th sample
# for i in range(0, num_samples, 5):  # Increment by 10
#     plt.plot(data[i, :, 0], data[i, :, 1], label=f'Sample {i+1}')
#
# plt.title('Plots from .npy File (Every 10th Sample)')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.legend()
# plt.show()

# Sample code to demonstrate the process with larger text font and correct step size.
# This code will not be executed here as it requires the actual .npy file.
import numpy as np
import matplotlib.pyplot as plt

# Load data from the .npy file
data = np.load('train.npy')  # Replace with your .npy file path

# Assuming the data is in the shape (num_samples, num_points, 2)
# Where the last dimension represents the x and y coordinates
# print(data.shape)
num_samples = data.shape[0]
print(num_samples)

plt.figure(figsize=(30, 10))

# Plot every 10th sample
for i in range(0, num_samples, 20):  # Increment by 10
    plt.plot(data[i, :, 0], data[i, :, 1], label=f'Sample {i+1}')

plt.title('UIUC airfoil Samples', fontsize=25)
plt.xlabel('X Coordinate', fontsize=25)
plt.ylabel('Y Coordinate', fontsize=25)
plt.legend(fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.show()

# NOTE: This is a placeholder code. The actual code execution requires the .npy file.

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load data from the .npy file
# file_path = '/mnt/data/test.npy'  # Adjust the path as needed
# data = np.load(file_path)
#
# # Check the shape of the data
# print(f"Data shape: {data.shape}")
#
# # Plot settings
# plt.figure(figsize=(20, 10))
#
# # Samples to plot
# samples_to_plot = [49, 99, 199, 299, 399, 499]
#
# # Plot the specified samples
# for i in samples_to_plot:
#     if i < data.shape[0]:  # Check if the sample index is within the range
#         plt.plot(data[i, :, 0], data[i, :, 1], label=f'Sample {i+1}')
#
# plt.title('Plots from test.npy File (Samples 50, 100, 200, 300, 400, 500)')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.legend()
# plt.show()
