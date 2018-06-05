"""
This script tests how tensorboar shows complex graphs
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Generating some sample data
train_X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
num_samples = train_X.shape[0]

# Parameters
learning_rate = 0.01
iterations = 50

### ------ ** Creating the graph ** -------

run_metadata = tf.RunMetadata()
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

# Placeholders - where the data can come into the graph
with tf.name_scope("Variables"):
    X = tf.placeholder(tf.float32, [None])
    Y = tf.placeholder(tf.float32, [None])


    # Creating the parameters - theta1 is the slope, theta0 is the intercept (y = theta0 + theta1*x)
    theta0 = tf.Variable(np.random.randn(), name='theta0')
    theta1 = tf.Variable(np.random.randn(), name='theta1')

# Creating the model: y = theta0 + theta1*x
model = tf.add(theta0, tf.multiply(theta1, X))

# Creating the cost function
cost_function = 0.5 * (1.0/num_samples) * tf.reduce_sum(tf.pow(model - Y, 2))

# Defining the method to do the minimisation of the cost function
with tf.name_scope("Optimiser"):
    optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

### -------- ** Initialising all the variables ** --------

init = tf.global_variables_initializer()

### -------- ** Starting the session ** ----------

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logdir="tensorboard", graph=sess.graph)

    for i in range(iterations):
        sess.run(optimiser, feed_dict={X: train_X, Y: train_Y}, options=options, run_metadata=run_metadata)

        if (i+1)%50 == 0:
            c = sess.run(cost_function, feed_dict={X: train_X, Y: train_Y})
            print("Step:", '%04d' % (i+1), "cost=", "{:.9f}".format(c), "Theta1=", sess.run(theta1), "Theta0=", sess.run(theta0))
            summary_writer.add_run_metadata(run_metadata=run_metadata, tag="%s" % i, global_step=None)

    slope = sess.run(theta1)
    intercept = sess.run(theta0)

