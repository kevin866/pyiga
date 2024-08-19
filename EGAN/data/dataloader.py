import os
import numpy as np

print("Current working directory:", os.getcwd())

train_data = np.load('EGAN/data/train.npy')
test_data = np.load('EGAN/data/test.npy')

print("Shape of train data:", train_data.shape)
print("Shape of test data:", test_data.shape)
