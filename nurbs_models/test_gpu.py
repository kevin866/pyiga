import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available. Using {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU")

# Create two random tensors on the device
matrix_size = 100000  # You can increase this to load the GPU more
A = torch.rand(matrix_size, matrix_size, device=device)
B = torch.rand(matrix_size, matrix_size, device=device)

# Perform matrix multiplication and time it
import time
start_time = time.time()

C = torch.matmul(A, B)

end_time = time.time()

print(f"Matrix multiplication completed in {end_time - start_time:.4f} seconds")
