"""
This test tries to understand what happens if you build a graph multiple times.
"""

import tensorflow as tf
import numpy as np

train_X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
num_samples = train_X.shape[0]
learning_rate = 0.01

graph = tf.Graph()


# Placeholders - where the data can come into the graph
X = tf.placeholder(tf.float32, [None])
Y = tf.placeholder(tf.float32, [None])

c = tf.Variable(np.random.randn(), name='intercept')
m = tf.Variable(np.random.randn(), name='slope')

# Creating the model: y = c + m*x
model = tf.add(c, tf.multiply(m, X))

# Creating the cost function
cost_function = 0.5 * (1.0 / num_samples) * tf.reduce_sum(tf.pow(model - Y, 2))

# Defining the method to do the minimisation of the cost function
optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(c))
print(c)
sess.run(optimiser, feed_dict={X: train_X, Y: train_Y})
print(sess.run(c))
# summary_writer = tf.summary.FileWriter(logdir="tensorboard", graph=sess.graph)

# Placeholders - where the data can come into the graph
X = tf.placeholder(tf.float32, [None])
Y = tf.placeholder(tf.float32, [None])

c = tf.Variable(np.random.randn(), name='intercept')
m = tf.Variable(np.random.randn(), name='slope')

# Creating the model: y = c + m*x
model = tf.add(c, tf.multiply(m, X))

# Creating the cost function
cost_function = 0.5 * (1.0 / num_samples) * tf.reduce_sum(tf.pow(model - Y, 2))

# Defining the method to do the minimisation of the cost function
optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(c))
print(c)
sess.run(optimiser, feed_dict={X: train_X, Y: train_Y})
print(sess.run(c))
summary_writer = tf.summary.FileWriter(logdir="tensorboard", graph=sess.graph)