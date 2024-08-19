import numpy as np
import matplotlib.pyplot as plt
import os

def plot_all_samples_in_one_figure(npy_file, num_samples, save_dir, display_plots=False):
    """
    Plots all specified samples from an .npy file in one figure.

    Parameters:
    npy_file (str): Path to the .npy file.
    num_samples (int): Number of samples to plot.
    save_dir (str): Directory where the plot will be saved.
    display_plots (bool): If True, displays the plot; otherwise, saves it.
    """
    # Load the data
    samples = np.load(npy_file)

    # Determine the layout of the subplots
    num_rows = int(np.ceil(10))
    print(num_rows)
    num_cols = int(np.ceil(3))

    # Create the figure
    plt.figure(figsize=(num_cols * 16, num_rows * 4))

    # Plot each sample in a subplot
    for i in range(min(num_samples, len(samples))):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.plot(samples[i, :, 0], samples[i, :, 1], marker='o', c='black', linewidth=0.1)  # Set color to black
        # plt.title(f'Sample {i+1}')
        # plt.xlabel('Dimension 1')
        # plt.ylabel('Dimension 2')
        plt.axis('off')
        plt.grid(True)

    # Adjust layout
    plt.subplots_adjust(wspace=0, hspace=1)

    if display_plots:
        plt.show()
    else:
        # Save the figure to the specified directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, 'all_samples.png'))
        plt.close()

# Example usage
npy_file = 'T.npy'  # Replace with your file path
num_samples = 30  # Number of samples you want to plot
save_dir = '/path/to/save/directory'  # Replace with your save directory
display_plots = True  # Set to True to display the plot, False to save it

plot_all_samples_in_one_figure(npy_file, num_samples, save_dir, display_plots)
