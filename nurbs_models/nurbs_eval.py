import numpy as np

def validate_attributes(x, types, conditions):
    if not isinstance(x, tuple(types)):
        raise ValueError(f"Expected types {types} but got {type(x)}")
    if "positive" in conditions and np.any(x <= 0):
        raise ValueError("All values must be positive.")
    if "integer" in conditions and not np.all(np.equal(np.mod(x, 1), 0)):
        raise ValueError("All values must be integers.")
    if "scalar" in conditions and np.isscalar(x) is False:
        raise ValueError("Expected scalar value.")
        
def basis_function(n, npts, u, t):
    """
    Evaluate the NURBS basis function at specified u.

    Parameters:
    n : int
        NURBS order (2 for linear, 3 for quadratic, 4 for cubic, etc.).
    npts : int
        Number of control points.
    u : float
        The point where the basis function is evaluated.
    t : list or np.ndarray
        Knot vector.

    Returns:
    N : np.ndarray
        Vector of size npts containing the value of the basis function at u.
    """
    
    nplusc = npts + n
    N = np.zeros(npts)
    # print(N)

    # First step: Initialize the basis function values (order 1)
    for i in range(npts):
        if t[i] <= u < t[i + 1]:
            N[i] = 1
        else:
            N[i] = 0
    if u == t[-1]:
        N[-1] = 1  # Handle special case when u equals the last knot
    # Recursively apply the Cox-de Boor recursion formula to get higher order basis functions

    for k in range(2, n+1):
        for i in range(nplusc - k):
            # print(i)
            d = 0
            
            if i<len(N):
                if N[i] != 0:
                    d = ((u - t[i]) * N[i]) / (t[i + k - 1] - t[i]) 
            e = 0 
            if i<len(N)-1:
                if N[i + 1] != 0:
                    e = ((t[i + k] - u) * N[i + 1]) / (t[i + k] - t[i + 1])
            if i<len(N):
                N[i] = d + e
    print(N)
    return N[:npts]


def nurbsfun(n, t, w, P, U=None):
    # input:
    # n: nurbs order
    # t: knot vector
    # w: weight vector
    # P: control points, usually 2 by m
    # U (optional):
    #    values where the NURBS is to be evaluated, or a positive
    #    integer to set the number of points to automatically allocate
    # Output: 
    #     C: points of Nurbs curve
    # Validate input
    validate_attributes(n, (int,), ['positive', 'integer', 'scalar'])
    if not np.all(np.diff(t) >= 0):
        raise ValueError("Knot vector values should be nondecreasing.")
    if P.shape[0] != 2:
        raise ValueError("P should have 2 rows representing control points.")
    
    nctrl = len(t) - n
    if P.shape[1] != nctrl:
        raise ValueError(f"Invalid number of control points, {P.shape[1]} given, {nctrl} required.")
    if len(w) != nctrl:
        raise ValueError(f"Invalid number of weights, {len(w)} given, {nctrl} required.")
    
    # Handle optional U input
    if U is None:
        U = np.linspace(t[n-1], t[-n], 10 * P.shape[1])
    elif isinstance(U, int) and U > 1:
        U = np.linspace(t[n-1], t[-n], U)
    
    totalU = len(U)
    nc = P.shape[1]
    N = np.zeros((totalU, nc))
    
    # Calculate basis functions
    for i in range(totalU):
        u = U[i]
        N[i, :] = basis_function(n, nc, u, t)
    
    # Calculate denominator of rational basis functions
    S = np.zeros(totalU)
    for i in range(totalU):
        S[i] = np.sum(N[i, :] * w)
    
    # Calculate rational basis functions
    R = np.zeros((totalU, nc))
    for i in range(totalU):
        if S[i] != 0:
            R[i, :] = (N[i, :] * w) / S[i]
        else:
            R[i, :] = 0
    
    # Calculate curve points
    C = np.zeros((2, totalU))
    for i in range(totalU):
        C[:, i] = np.dot(P, R[i, :])
    
    return C, N, R, S, U
