
## 1. Introduction:
NURBS are a mathematical representation that generalizes Bézier curves and surfaces. They are widely used in computer graphics and CAD systems for representing curves and surfaces. NURBS offer more flexibility than Bézier curves because of their non-uniform parameterization and the use of weights associated with control points.

## 2. Key Components:
### a. Knots:
Knots are a set of non-decreasing values that define how the control points influence the curve. In the `NURBSLayer`, the knot vector is computed using the method `_compute_open_knot_vector`. This method generates an open knot vector which means that the curve starts and ends at the first and last control points respectively.

### b. Control Points:
Control points define the shape of the curve. They are inputs to the layer and are denoted as `control_points`.

### c. Weights:
Each control point has an associated weight (`weights`). When the weight is greater than 1, it attracts the curve towards the control point, and when it's less than 1, it repels the curve away from the control point.

## 3. Basis Function:
The `basis_function` method computes the NURBS basis function. This function determines how much influence a control point has on the curve at a given parameter value `t`.

- If \( p = 0 \), the function is piecewise constant.
- For higher values of \( p \), the function provides a smooth blending of the control points' influence.

The function uses recursive evaluations based on the Cox-de Boor formula. The recursion is based on the degree `p`.

## 4. Forward Pass:
The `forward` method computes the NURBS curve given the input tensor, control points, and weights. Here's a breakdown:

### a. Intervals:
First, it generates intervals (`intvls`) using a neural network sub-module (`generate_intervals`). These intervals represent how the curve will be parameterized.

### b. Evaluation Points:
The cumulative sum of these intervals (`ub`) gives the parameter values at which the NURBS curve will be evaluated. They are clamped between 0 and 1.

### c. Compute Basis Functions:
For each control point, the NURBS basis function is computed at each evaluation point. This results in a tensor `N` that captures the influence of each control point at each evaluation point.

### d. Curve Computation:
The curve is computed as a weighted sum of the control points influenced by the basis functions. The weights are considered by multiplying them with the control points (`cp_w`). The curve (`dp`) is the result of the weighted control points' influence normalized by the sum of the weights.

The output of the layer is the computed curve `dp`, the evaluation points `ub`, and the intervals `intvls`.

## 5. Remarks:
- The layer uses an epsilon (`EPSILON`) value to prevent divisions by zero.
- The layer is designed to handle batches of input, control points, and weights. This is evident from the tensor shapes in the comments.

## Conclusion:
The `NURBSLayer` is a neural network layer that computes a NURBS curve based on input features, control points, and associated weights. It's a unique layer that combines traditional geometric modeling with deep learning methodologies.



### 1. NURBS Formulation:

A point \( P(u) \) on a NURBS curve is defined as:

\[
P(u) = \frac{\sum_{i=0}^{n} N_{i,p}(u) w_i P_i}{\sum_{i=0}^{n} N_{i,p}(u) w_i}
\]

Where:
- \( P_i \) are the control points.
- \( w_i \) are the weights associated with each control point.
- \( N_{i,p}(u) \) is the B-spline basis function of degree \( p \) for the \( i^{th} \) control point.
- \( n \) is the number of control points minus one (i.e., \( n = \text{number of control points} - 1 \)).

### 2. B-Spline Basis Function:

The basis function \( N_{i,p}(u) \) is defined recursively as:

\[
N_{i,0}(u) = \begin{cases} 
1 & \text{if } u_i \leq u < u_{i+1} \\
0 & \text{otherwise}
\end{cases}
\]

\[
N_{i,p}(u) = \frac{u - u_i}{u_{i+p} - u_i} N_{i,p-1}(u) + \frac{u_{i+p+1} - u}{u_{i+p+1} - u_{i+1}} N_{i+1,p-1}(u)
\]

Where:
- \( u \) is the parameter value.
- \( u_i \) are the knot values.
- \( p \) is the degree of the B-spline.

### 3. Forward Pass Computation:

Given an input tensor, the forward method computes the NURBS curve as follows:

1. **Interval Generation**:
   - Intervals are computed as:
     \[
     \text{intvls} = \text{Softmax}(\text{Linear}(\text{input}))
     \]
   - Cumulative sum of intervals gives parameter values:
     \[
     \text{ub} = \text{Cumsum}(\text{intvls})
     \]

2. **Basis Function Computation**:
   - For each control point and for each parameter value in \( \text{ub} \), compute the B-spline basis function \( N_{i,p}(u) \).

3. **Curve Computation**:
   - Compute weighted control points:
     \[
     \text{cp\_w} = \text{control\_points} \times \text{weights}
     \]
   - Compute the curve as:
     \[
     \text{dp} = \frac{\sum_{i=0}^{n} N_{i,p}(u) \times \text{cp\_w}_i}{\sum_{i=0}^{n} N_{i,p}(u) \times \text{weights}_i}
     \]

The output is the computed curve `dp`, the evaluation points `ub`, and the intervals `intvls`.

### Conclusion:
The `NURBSLayer` computes a NURBS curve by evaluating it at certain parameter values (`ub`) determined from the input. This is achieved through the combination of B-spline basis functions, control points, and their associated weights.