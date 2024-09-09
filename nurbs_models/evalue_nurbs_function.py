

import numpy as np
from scipy.interpolate import BSpline

def extract_bezier_curve(control_points, weights, knot_vector, degree, i):
    """
    Extracts the Bezier curve from the NURBS curve in the given knot interval [a, b].
    """
    # Extract control points and weights for the Bezier curve
    bezier_points = control_points[i:i+degree+1]
    bezier_weights = weights[i:i+degree+1]
    # Normalize the control points by the weights
    weighted_points = bezier_points * bezier_weights[:, np.newaxis]
    # Extract the knot interval [a, b]
    a, b = knot_vector[i], knot_vector[i+1]
    
    return BSpline(knot_vector[i:i+degree+3], weighted_points, degree), a, b

def find_parameter_value(nurbs_curve, control_points, weights, knot_vector, degree, X0):
    """
    Finds the parameter value corresponding to the given X0 in the NURBS curve.
    """
    num_intervals = len(knot_vector) - degree - 1
    
    for i in range(num_intervals):
        # Extract the Bezier curve and knot interval [a, b]
        bezier_curve, a, b = extract_bezier_curve(control_points, weights, knot_vector, degree, i)
        
        # Calculate the X-min and X-max for the current Bezier curve's control points
        Xmin, Xmax = np.min(bezier_curve.c[:, 0]), np.max(bezier_curve.c[:, 0])
        
        if X0 >= Xmin and X0 <= Xmax:
            t0, t1 = a, b
            epsilon = 1.0e-06
            
            while (t1 - t0) > epsilon:
                # Subdivide the Bezier curve at t=0.5
                t_mid = (t0 + t1) / 2
                B1, B2 = bezier_curve(t0), bezier_curve(t_mid), bezier_curve(t1)
                
                # Compute X-min and X-max for both subdivided curves
                Xmin1, Xmax1 = np.min(B1[:, 0]), np.max(B1[:, 0])
                Xmin2, Xmax2 = np.min(B2[:, 0]), np.max(B2[:, 0])
                
                if X0 >= Xmin1 and X0 <= Xmax1:
                    bezier_curve = B1
                    t1 = t_mid
                else:
                    bezier_curve = B2
                    t0 = t_mid
            
            # Return the parameter value corresponding to X0
            return (t0 + t1) / 2
    
    return None  # Return None if X0 is not found in any knot interval
