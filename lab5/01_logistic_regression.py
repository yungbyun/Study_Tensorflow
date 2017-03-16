from __future__ import print_function

import numpy as np
import tensorflow as tf

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

error = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
learning = optimizer.minimize(error)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(learning, feed_dict={X: x_data, Y: y_data})

    if step % 20 == 0:
        print(step, sess.run(error, feed_dict={X: x_data, Y: y_data}), sess.run(W))
