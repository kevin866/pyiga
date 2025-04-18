import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# # Superformula generation function
# def superformula(m, n1, n2, n3, a=1, b=1, num_points=500):
#     phi = np.linspace(0, 2 * np.pi, num_points)
#     r = (np.abs(np.cos(m * phi / 4) / a)**n2 + np.abs(np.sin(m * phi / 4) / b)**n3)**(-1 / n1)
#     x = r * np.cos(phi)
#     y = r * np.sin(phi)
#     return np.vstack((x, y)).T


# # Generate a dataset of superformula points
# superformula_params = [(m, n1, n2, n3) for m in range(1, 10) for n1 in range(1, 5) for n2 in range(1, 5) for n3 in range(1, 5)]
# superformula_points = [superformula(*params) for params in superformula_params]

def cal_c(r, a, L0):
    return np.sqrt(2)*np.sqrt(np.pi*(3+3*r**2+2*r)*a*L0)/(np.pi*(3+3*r**2+2*r))
def superformula(r, L0, n, a=0.5, d=1, num_res=100):
    theta = np.linspace(0, 2 * np.pi, num_res)
    c = cal_c(r,a,L0)
    # print(c)
    result = c*((1+r)-d*(-1)**((n+2)/2)*(r-1)*np.cos(n*theta))
    x = result * np.cos(theta)
    y = result * np.sin(theta)
    # return (x,y)
    return np.vstack((x, y)).T

# Generate a dataset of superformula points
superformula_params = [(round(r, 2), L0, n, (round(a, 2))) for r in np.arange(0.2, 0.9, 0.1).tolist() for L0 in np.arange(15, 35, 2.5).tolist()
                       for n in np.arange(2, 12, 2).tolist() for a in np.arange(0.2, 0.7, 0.1).tolist()]
# print(superformula_params)
superformula_points = np.array([superformula(*params) for params in superformula_params])
print(superformula_points.shape)
print(superformula_points[0,0,:])
np.random.shuffle(superformula_points)

train_size = 1200
test_size = 200

train_set = superformula_points[:train_size]
test_set = superformula_points[train_size:train_size + test_size]

np.save('kwtrain.npy', train_set)
np.save('kwtest.npy', test_set)

print("Train set shape:", train_set.shape)  
print("Test set shape:", test_set.shape)    