import tensorflow as tf
import numpy as np
from aglaia import symm_funct

xyzs_list = [[[0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0] ]]

Zs_list = [[7, 2, 1, 1]]

elements_list = [1, 2, 7]
element_pairs_list = [[1, 1], [2, 1], [7, 1], [7, 2]]


# Zs_list = [[7, 1, 1, 1]]
#
# elements_list = [7, 1]
# element_pairs_list = [[1, 1], [7, 1]]

# xyzs_list = [[[1., 2., 3.], [4., 5., 6.], [1, 1, 1], [2, 2, 2]], [[2, 4, 6], [8, 10, 12], [1, 1, 1], [2, 2, 2]],
#             [[3, 3, 3], [4, 4, 4], [1, 1, 1], [2, 2, 2]]]
# Zs_list = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
# elements = [1]

# radial_cutoff = 500.0
# angular_cutoff = 500.0
# radial_rs = [0.0, 0.1, 0.2]
# angular_rs = [0.0, 0.1, 0.2]
# theta_s = [3.0, 2.0]
# zeta = 3.0
# eta = 2.0

# The data
Zs = tf.constant(Zs_list)
xyzs = tf.constant(xyzs_list)
elements = tf.constant(elements_list)
element_pairs = tf.constant(element_pairs_list)

radial_cutoff = tf.constant(500.0, dtype=tf.float32)
angular_cutoff = tf.constant(500.0, dtype=tf.float32)
radial_rs = tf.constant([0.0, 0.1, 0.2], dtype=tf.float32)
angular_rs = tf.constant([0.0, 0.1, 0.2], dtype=tf.float32)
theta_s = tf.constant([3.0, 2.0], dtype=tf.float32)
zeta = tf.constant(3.0, dtype=tf.float32)
eta = tf.constant(2.0, dtype=tf.float32)
#
# n_atoms = Zs.get_shape().as_list()[1]
# n_samples = Zs.get_shape().as_list()[0]
# n_pairs = len(element_pairs_list)
# n_elements = elements.get_shape().as_list()[0]
# n_rs = angular_rs.get_shape().as_list()[0]
# n_thetas = theta_s.get_shape().as_list()[0]

# acsf = symm_funct.generate_parkhill_acsf(xyzs, Zs, elements, element_pairs)

pre_sum_ang = symm_funct.acsf_ang(xyzs, Zs, angular_cutoff, angular_rs, theta_s, zeta, eta)
# Then doing the sum based on the neighbrouing pair identity
ang_term = symm_funct.sum_ang(pre_sum_ang, Zs, element_pairs_list, angular_rs, theta_s)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
someting = sess.run(ang_term)
print(someting)




