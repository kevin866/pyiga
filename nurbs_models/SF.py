import numpy as np
import matplotlib.pyplot as plt


# Replace 'file_path.npy' with the path to your .npy file
# array = np.load('train.npy')
# print(array.shape)
# Now, 'array' contains the data from the .npy file
## (306, 192, 2) test size ## (1222, 192, 2) train size ##
# def superformula(m, n1, n2, n3, num_points=128):
#     """Generate coordinates for a superformula shape."""
#     theta = np.linspace(0, 2 * np.pi, num_points)
#     r = (np.abs(np.cos(m * theta / 4))**n2 + np.abs(np.sin(m * theta / 4))**n3)**(-1/n1)
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
#     return x, y

# # Parameters for Superformula I (m=3)
# s1_I, s2_I = 3, 5  # Example values for s1 and s2
# n1_I, n2_I, n3_I = s1_I, s1_I + s2_I, s1_I + s2_I
#
# # Parameters for Superformula II (m=4)
# s1_II, s2_II = 7, 2  # Example values for s1 and s2
# n1_II, n2_II, n3_II = s1_II, s1_II + s2_II, s1_II + s2_II
#
# # Generate coordinates
# x_I, y_I = superformula(m=3, n1=n1_I, n2=n2_I, n3=n3_I)
# x_II, y_II = superformula(m=4, n1=n1_II, n2=n2_II, n3=n3_II)
#
# # Plotting
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(x_I, y_I, label='Superformula I (m=3)')
# plt.title('Superformula I (m=3)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('equal')
#
# plt.subplot(1, 2, 2)
# plt.plot(x_II, y_II, label='Superformula II (m=4)')
# plt.title('Superformula II (m=4)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('equal')
#
# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Integrating the scaling down code into the provided code

def superformula(m, n1, n2, n3, num_points=192):
    """Generate coordinates for a superformula shape."""
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = (np.abs(np.cos(m * theta / 4))**n2 + np.abs(np.sin(m * theta / 4))**n3)**(-1/n1)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

num_samples = 500  # Total number of samples
num_points = 192   # Number of points per shape

# Initialize array to hold all coordinates
all_coordinates = np.zeros((num_samples, num_points, 2))

# Start values
s1_start, s2_start = 1, 1

# End values
s1_end, s2_end = 5, 20

for i in range(num_samples):
    # Gradually changing parameters
    s1 = s1_start + (s1_end - s1_start) * i / num_samples
    s2 = s2_start + (s2_end - s2_start) * i / num_samples
    n1, n2, n3 = s1, s1 + s2, s1 + s2

    # Generate coordinates for Superformula I (m=3)
    x, y = superformula(6, n1, n2, n3, num_points)

    # Translate the shape to start from (0,0)
    x_min = np.min(x)
    x -= x_min

    # Store translated coordinates
    all_coordinates[i, :, 0] = x
    all_coordinates[i, :, 1] = y

# Scaling down the data so that the maximum value is 1.00
max_abs_val = np.max(np.abs(all_coordinates))
all_coordinates /= max_abs_val

# Save the scaled array to a .npy file
np.save('./TrainS4.npy', all_coordinates)

# Path of the saved file
saved_file_path = './trainS3.npy'
saved_file_path




# Plotting all the generated shapes
plt.figure(figsize=(10, 10))
for i in range(num_samples):
    plt.plot(all_coordinates[i, :, 0], all_coordinates[i, :, 1], label=f'Shape {i+1}')
plt.title('Superformula Shapes')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.axis('equal')
plt.show()
