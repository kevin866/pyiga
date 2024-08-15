"""
BezierGAN for capturing the airfoil manifold

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
import tensorflow as tf

import sys 
sys.path.append('..')
from shape_plot import plot_grid


def preprocess(X):
    X = np.expand_dims(X, axis=-1)
    return X.astype(np.float32)

def postprocess(X):
    X = np.squeeze(X)
    return X


import tensorflow as tf

# Initialization and Building:
#
# The class BS2 is a custom Keras layer designed to handle B-Spline transformations.
# It takes two important parameters: n_control_points (number of control points) and degree (degree of the B-spline curve).
# The build method computes the open knot vector. This vector is constructed such that the first
# p + 1 # p+1 knots are 0 and the last p+1 knots are 1. The rest are uniformly spaced.
class BS2(tf.keras.layers.Layer):
    def __init__(self, n_control_points, degree, **kwargs):
        super(BS2, self).__init__(**kwargs)
        self.n_control_points = n_control_points
        self.degree = degree
        self.EPSILON = 1e-7

    def build(self, input_shape):
        # Compute the open knot vector
        # First p+1 knots are 0, last p+1 knots are 1, the rest are uniformly spaced.
        self.knots = [0.0] * (self.degree + 1) + \
                     [i / (self.n_control_points - self.degree) for i in
                      range(1, self.n_control_points - self.degree)] + \
                     [1.0] * (self.degree + 1)
        self.knots = tf.constant(self.knots, dtype=tf.float32)

    def call(self, inputs):
        cp, w, ub = inputs

        # B-spline basis function
        def basis_function(t, i, p, knots):
            if p == 0:
                return tf.cast(tf.logical_and(knots[i] <= t, t < knots[i + 1]), tf.float32)
            else:
                A = ((t - knots[i]) / (knots[i + p] - knots[i] + self.EPSILON)) * basis_function(t, i, p - 1, knots)
                B = ((knots[i + p + 1] - t) / (knots[i + p + 1] - knots[i + 1] + self.EPSILON)) * basis_function(t,
                                                                                                                 i + 1,
                                                                                                                 p - 1,
                                                                                                                 knots)
                return A + B

        # Compute values of basis functions at data points
        N = []
        for j in range(self.n_control_points):
            N_j = basis_function(ub, j, self.degree, self.knots)
            N.append(N_j)
        N = tf.stack(N, axis=-1)  # batch_size x n_data_points x n_control_points
        print('nfasdg',N.shape)
        N = tf.reshape(N, [-1, 192, int(self.n_control_points)])
        # Compute data points
        cp_w = tf.multiply(cp, w)
        print(N.shape, w.shape, cp_w.shape)
        dp = tf.matmul(N, cp_w)  # batch_size x n_data_points x 2
        # N_w = tf.multiply(N, w)
        N_w = tf.matmul(N, w)  # batch_size x n_data_points x 1
        # print(N_w.shape)(?, 192, 1)
        dp = tf.divide(dp, N_w + self.EPSILON)  # batch_size x n_data_points x 2
        dp = tf.expand_dims(dp, axis=-1)  # batch_size x n_data_points x 2 x 1

        return dp


# Test the B-spline layer

# dp_bspline.shape


# Define the B-spline basis function
# def vectorized_bspline_basis(knot_vector, k, ub_expanded):
#     # Base case k = 0
#     if k == 0:
#         condition = (knot_vector[:-1] <= ub_expanded) & (ub_expanded < knot_vector[1:])
#         return tf.cast(condition, tf.float32)
#
#     # Recursive case
#     denom1 = knot_vector[k::-1] - knot_vector[:-k]
#
#     denom2 = knot_vector[k + 1::-1] - knot_vector[1:-k - 1]
#
#     term1 = tf.zeros_like(ub_expanded)
#     term2 = tf.zeros_like(ub_expanded)
#
#     valid_denom1 = tf.where(denom1 > 0)
#     valid_denom2 = tf.where(denom2 > 0)
#
#     term1 = tf.tensor_scatter_nd_update(
#         term1,
#         valid_denom1,
#         ((ub_expanded - knot_vector[:-k]) / denom1) * vectorized_bspline_basis(knot_vector, k - 1, ub_expanded)
#     )
#
#     term2 = tf.tensor_scatter_nd_update(
#         term2,
#         valid_denom2,
#         ((knot_vector[k + 1:] - ub_expanded) / denom2) * vectorized_bspline_basis(knot_vector, k - 1, ub_expanded)
#     )
#
#     return term1 + term2
#
#
# def compute_bspline(ub, cp, w, knot_vector, k):
#     # Broaden the parameter values across control points
#     ub_expanded = tf.expand_dims(ub, -2)  # [1, n_data_points, 1, 1]
#
#     # Broadcast control points and weights to the shape of [batch_size, n_data_points, bezier_degree+1, 2/1]
#     cp_broadcasted = tf.expand_dims(cp, 1)  # [batch_size, 1, bezier_degree+1, 2]
#     w_broadcasted = tf.expand_dims(w, 1)  # [batch_size, 1, bezier_degree+1, 1]
#
#     # Compute the basis function values for all control points and parameter values
#     # Assuming a vectorized version of bspline_basis that computes for all control points and parameter values
#     basis_values = vectorized_bspline_basis(knot_vector, k, ub_expanded)  # [1, n_data_points, bezier_degree+1]
#
#     # Compute the weighted sum of control points using the basis function values
#     weighted_cp = cp_broadcasted * w_broadcasted * basis_values  # Broadcasting multiplication
#     dp_num = tf.reduce_sum(weighted_cp, axis=2)  # Summing along the control points axis
#
#     # Normalize the computed data points by the sum of basis function values
#     basis_sum = tf.reduce_sum(basis_values * w_broadcasted, axis=2)
#     dp = dp_num / tf.expand_dims(basis_sum, -1)
#
#     dp = tf.expand_dims(dp, -1)  # Add the last dimension
#     return dp
#
#
# # Test the functions with dummy values
# batch_size = 2
# n_data_points = 192
# bezier_degree = 3
# k = 3  # Degree of B-spline
# num_control_points = bezier_degree + 1
#
# # Dummy values
# ub = tf.random.uniform([1, n_data_points, 1], minval=0, maxval=1)
# cp = tf.random.normal([batch_size, num_control_points, 2])
# w = tf.random.normal([batch_size, num_control_points, 1])




def bspline_basis2(i, p, u, U):
    """
    Compute the B-spline basis function value.

    i: control point index
    p: degree of the B-spline
    u: parameter value
    U: knot vector
    """

    if p == 0:
        return tf.cast(tf.math.logical_and(u >= U[i], u < U[i + 1]), tf.float32)

    first_term = 0
    if (U[i + p] - U[i]) != 0:
        first_term = ((u - U[i]) / (U[i + p] - U[i])) * bspline_basis2(i, p - 1, u, U)

    second_term = 0
    if (U[i + p + 1] - U[i + 1]) != 0:
        second_term = ((U[i + p + 1] - u) / (U[i + p + 1] - U[i + 1])) * bspline_basis2(i + 1, p - 1, u, U)

    return first_term + second_term


def b_spline_layer2(ub, cp, w, U, p):
    """
    B-spline layer.

    ub: parameter values
    cp: control points
    w: weights
    U: knot vector
    p: degree of the B-spline
    """

    batch_size, n_data_points, _ = ub.shape
    n_control_points = cp.shape[1]

    N = []
    for i in range(n_control_points):
        N.append(bspline_basis2(i, p, ub, U))
    N = tf.stack(N, axis=-1)  # batch_size x n_data_points x n_control_points

    cp_w = tf.multiply(cp, w)

    dp = tf.matmul(N, cp_w)  # batch_size x n_data_points x 2
    N_w = tf.matmul(N, w)  # batch_size x n_data_points x 1
    dp = tf.div(dp, N_w)  # batch_size x n_data_points x 2
    dp = tf.expand_dims(dp, axis=-1)  # batch_size x n_data_points x 2 x 1

    return dp


# # Test data
# batch_size = 2
# n_data_points = 100
# bezier_degree = 3
# cp = tf.random.normal([batch_size, bezier_degree + 1, 2])
# w = tf.random.normal([batch_size, bezier_degree + 1, 1])
# ub = tf.linspace(0.0, 1.0, n_data_points)
# ub = tf.reshape(ub, [1, n_data_points, 1])
# ub = tf.tile(ub, [batch_size, 1, 1])
#
# # Define a simple knot vector for the B-spline

# dp.shape





# # Convert Bezier to B-spline with sample control points and weights
# c, t, k = bezier_to_bspline(sample_cp, sample_w, sample_bezier_degree)
#
# # Create B-spline representation
# bspline = BSpline(t, c, k)
#
# c, t, k



class BSplineLayer(tf.keras.layers.Layer):
    def __init__(self, bezier_degree, n_data_points, bounds=(0.0, 1.0), **kwargs):
        super(BSplineLayer, self).__init__(**kwargs)
        self.bezier_degree = bezier_degree
        self.n_data_points = n_data_points
        self.num_control_points = bezier_degree + 1

        # Adjusting the knot vector generation for uniform B-spline
        self.knots = tf.concat([
            tf.zeros([self.bezier_degree], dtype=tf.float32),
            tf.linspace(bounds[0], bounds[1], num=self.num_control_points - self.bezier_degree + 1),
            tf.ones([self.bezier_degree], dtype=tf.float32)
        ], axis=0)

    def _compute_bspline_basis(self, u, degree):
        # print(u)
        if degree == 0:
            return tf.cast(tf.logical_and(u >= self.knots[:-1], u < self.knots[1:]), tf.float32)
        # print(u)
        basis_n1 = self._compute_bspline_basis(u, degree - 1)
        # print(basis_n1)

        # Adjust front to align with basis_n1
        front_terms = (u - self.knots[:-degree - 1]) / (self.knots[degree:-1] - self.knots[:-degree - 1] + 1e-6)
        front = front_terms[:, :, :basis_n1.shape[-1] - 1]

        # Adjust back to align with basis_n1
        numerator = self.knots[degree + 1:-1] - u
        denominator = self.knots[degree + 1:-1] - self.knots[2:-degree] + 1e-6
        back_terms = numerator / denominator
        # Padding back to match the shape of front
        zeros_padding = tf.zeros_like(back_terms[:, :, :1])
        back = tf.concat([back_terms, zeros_padding], axis=-1)

        # print("front shape:", front.shape)
        # print("basis_n1[:, :, :-1] shape:", basis_n1[:, :, :-1].shape)
        # print("back shape:", back.shape)
        # print("basis_n1[:, :, 1:] shape:", basis_n1[:, :, 1:].shape)

        return front * basis_n1[:, :, :-1] + back * basis_n1[:, :, 1:]

    def call(self, ub, cp, w):
        print(ub)
        # print(inputs)
        # ub, cp, w = inputs
        # print(ub)

        # Compute the B-spline basis for all control points
        bs = self._compute_bspline_basis(ub, self.bezier_degree)
        # print(bs)

        w_reshaped = tf.squeeze(w, axis=-1)  # Shape: (?, 32, 1)
        w_reshaped_expanded = tf.expand_dims(w_reshaped, axis=-1)  # Shape: (?, 32, 1, 1)

        cp_w = cp * w_reshaped_expanded  # Shape: (?, 32, 2)

        dp = tf.matmul(bs, cp_w)  # Shape: (?, 192, 2)

        bs_w = tf.matmul(bs, w_reshaped)  # Shape: (?, 192, 1)
        bs_w = tf.reduce_sum(bs_w, axis=2)  # Shape: (?, 192)

        bs_w_expanded = tf.expand_dims(bs_w, axis=-1)  # Add an extra dimension to match dp: Shape: (?, 192, 1)

        # Replicate the last dimension to match dp shape
        bs_w_expanded = tf.concat([bs_w_expanded, bs_w_expanded], axis=2)  # Shape: (?, 192, 2)

        dp = dp / (bs_w_expanded + 1e-6)  # To avoid division by zero

        # Add the last dimension to match the desired output shape
        dp = tf.expand_dims(dp, axis=-1)  # Shape: (?, 192, 2, 1)

        return dp


# Usage:
# bspline_layer = BSplineLayer(bezier_degree=31, n_data_points=192)
# dp = bspline_layer([ub, cp, w])
import tensorflow as tf

# Constants
EPSILON = 1e-7


# Forming the Open Uniform (Clamped) Knot Vector
def create_final_open_uniform_knot_vector(n, k):
    knot_vector = [0.0] * (k + 1)

    # Adjust the internal knots
    internal_knots_count = n - k
    for i in range(1, internal_knots_count + 1):
        knot_vector.append(i / (internal_knots_count + 1))

    knot_vector.extend([1.0] * (k + 1))
    return knot_vector


# B-spline basis functions using Cox-de Boor recursion
def bspline_basis(u, degree, knots):
    n = len(knots) - degree - 1
    basis = [tf.cast((u >= knots[i]) & (u < knots[i + 1]), tf.float32) for i in range(n)]

    for p in range(1, degree + 1):
        for i in range(n - p):
            left = 0.0
            right = 0.0
            if knots[i + p] - knots[i] != 0:
                left = ((u - knots[i]) / (knots[i + p] - knots[i])) * basis[i]
            if knots[i + p + 1] - knots[i + 1] != 0:
                right = ((knots[i + p + 1] - u) / (knots[i + p + 1] - knots[i + 1])) * basis[i + 1]
            basis[i] = left + right
        del basis[-1]  # Remove the last basis function

    return tf.stack(basis, axis=-1)


# B-spline layer to compute curve points
def bspline_layer0(ub, cp, w, degree, knots):
    ub_tile = tf.tile(ub, [1, 1, bezier_degree + 1])
    bs = bspline_basis(ub_tile, degree, knots)

    # Pad the bs tensor along its last dimension
    bs = tf.reshape(bs, [-1, 192, 32, 32])

    # Compute curve points
    cp_w = tf.multiply(cp, w)
    dp = tf.matmul(bs, cp_w)
    # print(cp_w.shape,dp.shape,bs.shape,w.shape)(?, 32, 2) (?, 192, 32, 2) (?, 192, 32, 32) (?, 32, 1)
    bs_w = tf.matmul(bs, w)
    dp = tf.divide(dp, bs_w + EPSILON)
    dp = tf.expand_dims(dp, axis=-1)

    return dp


# Parameters
bezier_degree = 31
k = 3  # Degree of B-spline
knot_vector = create_final_open_uniform_knot_vector(bezier_degree + 1, k)


class GAN(object):
    
    def __init__(self, latent_dim=4, noise_dim=10, n_points=192, bezier_degree=31, bounds=(0.0, 1.0)):

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        
        self.X_shape = (n_points, 2, 1)
        self.bezier_degree = bezier_degree
        self.bounds = bounds

        
    def generator(self, c, z, reuse=tf.AUTO_REUSE, training=True):

        depth_cpw = 32*8
        bezier_degree= int(self.bezier_degree)
        dim_cpw = int((self.bezier_degree+1)/8)
        print('dim', dim_cpw)
        kernel_size = (4,3)
        n_control_points = 64
#        noise_std = 0.01
        
        with tf.variable_scope('Generator', reuse=reuse):
                
            if self.noise_dim == 0:
                cz = c
            else:
                cz = tf.concat([c, z], axis=-1)
            
            cpw = tf.layers.dense(cz, 1024)
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
            # print('cpw', cpw.shape)
    
            cpw = tf.layers.dense(cpw, dim_cpw*3*depth_cpw)
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
            cpw = tf.reshape(cpw, (-1, dim_cpw, 3, depth_cpw))
    
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/2), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
#            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
#             print('cpw', cpw.shape)
            
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/4), kernel_size, strides=(2,1), padding='same')
            # print('cpw1', cpw.shape)
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
#            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
#             print(int(depth_cpw/8))
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/8), kernel_size, strides=(2,1), padding='same')
            # print('cpw1', cpw.shape)
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
#            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)

            
            # Control points
            cp = tf.layers.conv2d(cpw, 1, (1,2), padding='valid') # batch_size x (bezier_degree+1) x 2 x 1
            cp = tf.nn.tanh(cp)
            cp = tf.squeeze(cp, axis=-1, name='control_point') # batch_size x (bezier_degree+1) x 2
            cp = tf.keras.layers.UpSampling1D(size=2)(cp)

            # Apply a Conv1D layer to learn features from the upsampled tensor
            cp = tf.keras.layers.Conv1D(filters=2, kernel_size=3, padding='same', activation='relu')(cp)
            print('cp',cp.shape)
            
            # Weights
            w = tf.layers.conv2d(cpw, 1, (1,3), padding='valid')
            # print('w0',w.shape)
            w = tf.nn.sigmoid(w) # batch_size x (bezier_degree+1) x 1 x 1
            # print('w1',w.shape)
            w = tf.squeeze(w, axis=-1, name='weight') # batch_size x (bezier_degree+1) x 1
            # print('w', w.shape)
            w = tf.keras.layers.UpSampling1D(size=2)(w)

            # Apply a Conv1D layer to learn features from the upsampled tensor
            w = tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='same', activation='relu')(w)

            # print('ww',w.shape)
            
            # Parameters at data points
            db = tf.layers.dense(cz, 1024)
            db = tf.layers.batch_normalization(db, momentum=0.9)#, training=training)
            db = tf.nn.leaky_relu(db, alpha=0.2)
            
            db = tf.layers.dense(db, 256)
            db = tf.layers.batch_normalization(db, momentum=0.9)#, training=training)
            db = tf.nn.leaky_relu(db, alpha=0.2)
            
            db = tf.layers.dense(db, self.X_shape[0]-1)
            db = tf.nn.softmax(db) # batch_size x (n_data_points-1)
            # print('db',db.shape)
            
#            db = tf.random_gamma([tf.shape(cz)[0], self.X_shape[0]-1], alpha=100, beta=100)
#            db = tf.nn.softmax(db) # batch_size x (n_data_points-1)
            
            ub = tf.pad(db, [[0,0],[1,0]], constant_values=0) # batch_size x n_data_points
            ub = tf.cumsum(ub, axis=1)
            ub = tf.minimum(ub, 1)
            ub = tf.expand_dims(ub, axis=-1) # 1 x n_data_points x 1
            # print('ub', ub.shape)
            # print('v',ub)
            # Step 1: Create an instance
            # bspline_layer = BSplineLayer(bezier_degree=31, n_data_points=192)
            #
            # # Step 2: Call the instance with data
            # U = [0.0] * (self.bezier_degree + 1) + [1.0] * (self.bezier_degree + 1)
            # U = tf.constant(U, dtype=tf.float32)
            # #
            # dp = b_spline_layer2(ub, cp, w, U, bezier_degree)

            bspline_layer = BS2(n_control_points, degree=3)
            dp = bspline_layer(
                [cp,w,ub])
            dp = tf.reshape(dp, [-1, 192, 2, 1])

            # c, t, k = bezier_to_bspline(cp, w, self.bezier_degree)
            # n_data_points = 192
            # bezier_degree = 3
            # k = 3  # Degree of B-spline
            # num_control_points = bezier_degree + 1
            # knot_vector = tf.linspace(0.0, 1.0, num_control_points + k + 1)
            #
            # dp = compute_bspline(ub, cp, w, knot_vector, k)
            # Bezier layer
            # Compute values of basis functions at data points
            # num_control_points = self.bezier_degree + 1
            # lbs = tf.tile(ub, [1, 1, num_control_points]) # batch_size x n_data_points x n_control_points
            # pw1 = tf.range(0, num_control_points, dtype=tf.float32)
            # pw1 = tf.reshape(pw1, [1, 1, -1]) # 1 x 1 x n_control_points
            # pw2 = tf.reverse(pw1, axis=[-1])
            # lbs = tf.add(tf.multiply(pw1, tf.log(lbs+EPSILON)), tf.multiply(pw2, tf.log(1-lbs+EPSILON))) # batch_size x n_data_points x n_control_points
            # lc = tf.add(tf.lgamma(pw1+1), tf.lgamma(pw2+1))
            # lc = tf.subtract(tf.lgamma(tf.cast(num_control_points, dtype=tf.float32)), lc) # 1 x 1 x n_control_points
            # lbs = tf.add(lbs, lc) # batch_size x n_data_points x n_control_points
            # bs = tf.exp(lbs)
            # # Compute data points
            # cp_w = tf.multiply(cp, w)
            # dp = tf.matmul(bs, cp_w) # batch_size x n_data_points x 2
            # bs_w = tf.matmul(bs, w) # batch_size x n_data_points x 1
            # dp = tf.div(dp, bs_w) # batch_size x n_data_points x 2
            # dp = tf.expand_dims(dp, axis=-1, name='fake_image') # batch_size x n_data_points x 2 x 1
            # dp=bspline_layer0(ub,cp,w,3,knot_vector)
            # dp = tf.squeeze(dp, axis=-1)
            # # dp = tf.layers.conv2d(dp, filters=1, kernel_size=1, strides=1, padding='same')(?, 192, 2, 1)
            # dp = tf.layers.conv2d(dp, filters=1, kernel_size=(1,1), strides=(1,1), padding='SAME')
            # dp = tf.layers.max_pooling2d(dp, pool_size=(1, 16), strides=(1, 16), padding='SAME')
            print(dp.shape)
            
            return dp, cp, w, ub, db
        
    def discriminator(self, x, reuse=tf.AUTO_REUSE, training=True):
        
        depth = 64
        dropout = 0.4
        kernel_size = (4,2)
        
        with tf.variable_scope('Discriminator', reuse=reuse):

            x = tf.layers.conv2d(x, depth*1, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*2, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*4, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*8, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*16, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*32, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.flatten(x)
            # print('x',x.shape)
            x = tf.layers.dense(x, 1024)
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            d = tf.layers.dense(x, 1)
            
            q = tf.layers.dense(x, 128)
            q = tf.layers.batch_normalization(q, momentum=0.9)#, training=training)
            q = tf.nn.leaky_relu(q, alpha=0.2)
            q_mean = tf.layers.dense(q, self.latent_dim)
            q_logstd = tf.layers.dense(q, self.latent_dim)
            q_logstd = tf.maximum(q_logstd, -16)
            # Reshape to batch_size x 1 x latent_dim
            q_mean = tf.reshape(q_mean, (-1, 1, self.latent_dim))
            q_logstd = tf.reshape(q_logstd, (-1, 1, self.latent_dim))
            q = tf.concat([q_mean, q_logstd], axis=1, name='predicted_latent') # batch_size x 2 x latent_dim
            
            return d, q
        
    def train(self, X_train, train_steps=2000, batch_size=256, save_interval=0, directory='.'):
            
        X_train = preprocess(X_train)
        
        # Inputs
        self.x = tf.placeholder(tf.float32, shape=(None,)+self.X_shape, name='real_image')
        self.c = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name='latent_code')
        self.z = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name='noise')
        
        # Targets
        q_target = tf.placeholder(tf.float32, shape=[None, self.latent_dim])
        
        # Outputs
        d_real, _ = self.discriminator(self.x)
        x_fake_train, cp_train, w_train, ub_train, db_train = self.generator(self.c, self.z)
        d_fake, q_fake_train = self.discriminator(x_fake_train)
        
        self.x_fake_test, self.cp, self.w, ub, db = self.generator(self.c, self.z, training=False)
        _, self.q_test = self.discriminator(self.x, training=False)
        
        # Losses
        # Cross entropy losses for D
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        # Cross entropy losses for G
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        # Regularization for w, cp, a, and b
        r_w_loss = tf.reduce_mean(w_train[:,1:-1], axis=[1,2])
        cp_dist = tf.norm(cp_train[:,1:]-cp_train[:,:-1], axis=-1)
        r_cp_loss = tf.reduce_mean(cp_dist, axis=-1)
        r_cp_loss1 = tf.reduce_max(cp_dist, axis=-1)
        ends = cp_train[:,0] - cp_train[:,-1]
        r_ends_loss = tf.norm(ends, axis=-1) + tf.maximum(0.0, -10*ends[:,1])
        r_db_loss = tf.reduce_mean(db_train*tf.log(db_train), axis=-1)
        r_loss = r_w_loss + r_cp_loss + 0*r_cp_loss1 + r_ends_loss + 0*r_db_loss
        r_loss = tf.reduce_mean(r_loss)
        # Gaussian loss for Q
        q_mean = q_fake_train[:, 0, :]
        q_logstd = q_fake_train[:, 1, :]
        epsilon = (q_target - q_mean) / (tf.exp(q_logstd) + EPSILON)
        q_loss = q_logstd + 0.5 * tf.square(epsilon)
        q_loss = tf.reduce_mean(q_loss)
        
        # Optimizers
        d_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)
        
        # Generator variables
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        # Discriminator variables
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        
        # Training operations
        d_train_real = d_optimizer.minimize(d_loss_real, var_list=dis_vars)
        d_train_fake = d_optimizer.minimize(d_loss_fake + q_loss, var_list=dis_vars)
        g_train = g_optimizer.minimize(g_loss + 10*r_loss + q_loss, var_list=gen_vars)
        
#        def clip_gradient(optimizer, loss, var_list):
#            grads_and_vars = optimizer.compute_gradients(loss, var_list)
#            clipped_grads_and_vars = [(grad, var) if grad is None else 
#                                      (tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
#            train_op = optimizer.apply_gradients(clipped_grads_and_vars)
#            return train_op
#        
#        d_train_real = clip_gradient(d_optimizer, d_loss_real, dis_vars)
#        d_train_fake = clip_gradient(d_optimizer, d_loss_fake + q_loss, dis_vars)
#        g_train = clip_gradient(g_optimizer, g_loss + q_loss, gen_vars)
        
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        
        # Create summaries to monitor losses
        tf.summary.scalar('D_loss_for_real', d_loss_real)
        tf.summary.scalar('D_loss_for_fake', d_loss_fake)
        tf.summary.scalar('G_loss', g_loss)
        tf.summary.scalar('R_loss', r_loss)
        tf.summary.scalar('Q_loss', q_loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Start training
        self.sess = tf.Session()
        
        # Run the initializer
        self.sess.run(init)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('{}/logs'.format(directory), graph=self.sess.graph)
    
        for t in range(train_steps):
    
            ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
            X_real = X_train[ind]
            _, dlr = self.sess.run([d_train_real, d_loss_real], feed_dict={self.x: X_real})
            y_latent = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(batch_size, self.latent_dim))
            noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
            X_fake = self.sess.run(self.x_fake_test, feed_dict={self.c: y_latent, self.z: noise})

            if np.any(np.isnan(X_fake)):
                ind = np.any(np.isnan(X_fake), axis=(1,2,3))
                print(self.sess.run(ub, feed_dict={self.c: y_latent, self.z: noise})[ind])
                assert not np.any(np.isnan(X_fake))
                
            _, dlf, qdl = self.sess.run([d_train_fake, d_loss_fake, q_loss],
                                        feed_dict={x_fake_train: X_fake, q_target: y_latent})
                
            y_latent = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(batch_size, self.latent_dim))
            noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
            
            _, gl, rl, qgl = self.sess.run([g_train, g_loss, r_loss, q_loss],
                                           feed_dict={self.c: y_latent, self.z: noise, q_target: y_latent})
            
            summary_str = self.sess.run(merged_summary_op, feed_dict={self.x: X_real, x_fake_train: X_fake,
                                                                      self.c: y_latent, self.z: noise, q_target: y_latent})
            
            summary_writer.add_summary(summary_str, t+1)
            
            # Show messages
            log_mesg = "%d: [D] real %f fake %f q %f" % (t+1, dlr, dlf, qdl)
            log_mesg = "%s  [G] fake %f reg %f q %f" % (log_mesg, gl, rl, qgl)
            print(log_mesg)
            
            if save_interval>0 and (t+1)%save_interval==0:
                
#                from matplotlib import pyplot as plt
#                
#                ub_batch, db_batch = self.sess.run([ub, db], feed_dict={self.c: y_latent, self.z: noise})
#                
#                xx = np.linspace(0, 1, self.X_shape[0])
#                plt.figure()
#                for u in np.squeeze(ub_batch):
#                    plt.plot(xx, u)
#                plt.savefig('{}/ub.svg'.format(directory))
#                
#                plt.figure()
#                for d in db_batch:
#                    plt.plot(xx[:-1], d)
#                plt.savefig('{}/db.svg'.format(directory))
                
                # Save the variables to disk.
                save_path = saver.save(self.sess, '{}/model'.format(directory))
                print('Model saved in path: %s' % save_path)
                print('Plotting results ...')
                plot_grid(5, gen_func=self.synthesize, d=self.latent_dim, bounds=self.bounds,
                          scale=.95, scatter=True, s=1, alpha=.7, fname='{}/synthesized'.format(directory))
                
        # Save the final variables to disk.
        save_path = saver.save(self.sess, '{}/model'.format(directory))
        print('Model saved in path: %s' % save_path)
                    
    def restore(self, directory='.'):
        
        self.sess = tf.Session()
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph('{}/model.meta'.format(directory))
        saver.restore(self.sess, tf.train.latest_checkpoint('{}/'.format(directory)))
        
        # Access and create placeholders variables            
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name('real_image:0')
        self.c = graph.get_tensor_by_name('latent_code:0')
        self.z = graph.get_tensor_by_name('noise:0')
        self.x_fake_test = graph.get_tensor_by_name('Generator_1/fake_image:0')
        self.cp = graph.get_tensor_by_name('Generator_1/control_point:0')
        self.w = graph.get_tensor_by_name('Generator_1/weight:0')
        self.q_test = graph.get_tensor_by_name('Discriminator_2/predicted_latent:0')

    def synthesize(self, latent, noise=None, return_cp=False):
        if isinstance(latent, int):
            N = latent
            latent = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(N, self.latent_dim))
            noise = np.random.normal(scale=0.5, size=(N, self.noise_dim))
            X, P, W = self.sess.run([self.x_fake_test, self.cp, self.w], feed_dict={self.c: latent, self.z: noise})
        else:
            N = latent.shape[0]
            if noise is None:
                noise = np.zeros((N, self.noise_dim))
            X, P, W = self.sess.run([self.x_fake_test, self.cp, self.w], feed_dict={self.c: latent, self.z: noise})
        if return_cp:
            return postprocess(X), postprocess(P), postprocess(W)
        else:
            return postprocess(X)
    
    def embed(self, X):
        latent = self.sess.run(self.q_test, feed_dict={self.x: X})
        return latent[:,0,:]
    
    