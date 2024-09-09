import numpy as np
import matplotlib.pyplot as plt
from nurbs_eval import nurbsfun

# Import the previously defined nurbsfun function
# Import the previously defined basis_function function

# Control points
P = np.array([[5, 2, 2, 6, 10, 10, 7], 
              [1, 3, 8, 10, 8, 3, 1]])

n = 3  # Curve's degree n = p + 1 where p is the degree of the curve

# Knot vector
t = [0, 0, 0, 0.1, 0.2, 0.5, 0.7, 1, 1, 1]

# Weight vector
w = [1, 1, 1, 1, 1, 1, 1]

# Call the NURBS function
C, N, R, S, U = nurbsfun(n, t, w, P)

# Plot the control points and NURBS curve
plt.figure()
plt.xlim([0, 12])
plt.ylim([0, 12])

# Plot control points
plt.plot(P[0, :], P[1, :], 'bo', label="Control Points")
plt.plot(P[0, :], P[1, :], 'g-', label="Control Polygon")
print(C)
# Plot NURBS curve
plt.plot(C[0, :], C[1, :], 'r-', label="NURBS Curve")

# # Changing weights
# w = [1, 1, 1, 3, 1, 2, 0]
# C, N, R, S, U = nurbsfun(n, t, w, P)
# plt.plot(C[0, :], C[1, :], 'b-', label="Modified Weights 1")

# # Changing weights again
# w = [1, 1, 3, 1, 4, 1, 1]
# C, N, R, S, U = nurbsfun(n, t, w, P)
# plt.plot(C[0, :], C[1, :], 'm-', label="Modified Weights 2")

# Display the legend and show the plot
plt.legend()
plt.show()
