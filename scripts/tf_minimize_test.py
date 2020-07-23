"""Tensorflow minimize example

"""

import numpy as np
import tensorflow as tf


# Create variables and constants
init_val = np.random.rand(5, 1)
init_val = init_val / np.linalg.norm(init_val)
p = tf.Variable(initial_value=init_val)
mu_ab = tf.constant(np.random.rand(5, 1))
mu_ba = tf.constant(np.random.rand(5, 1))
mu_a = tf.constant(np.random.rand(5, 1))
mu_b = tf.constant(np.random.rand(5, 1))


# Define loss
def loss():
    d_start = (tf.transpose(p) @ (mu_a - mu_b))**2
    d_mid = (tf.transpose(p) @ (mu_ab - mu_ba))**2
    l2 = tf.transpose(p) @ p
    return d_start - d_mid + l2


# Create optimizer object and minimize
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
epochs = 20
J = []
for _ in range(epochs):
    opt.minimize(loss, var_list=[p])
    J.append(loss().numpy())

print(J)
