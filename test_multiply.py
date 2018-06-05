import tensorflow as tf

a = tf.Variable(tf.ones(shape=[5, 3, 2], dtype=tf.int32))
b = tf.multiply(tf.Variable(tf.ones(shape=[2, 4], dtype=tf.int32)), tf.constant(2, dtype=tf.int32))
c = tf.tensordot(a, b, axes=1)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

c_np = sess.run(c)
print(c_np.shape)