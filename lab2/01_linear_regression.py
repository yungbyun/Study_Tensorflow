from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plot

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W * x_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

costs = []
weights = []
bs = []

for step in range(2001):
    sess.run(train)
    if step % 40 == 0:
        val_c = sess.run(cost)
        val_w = sess.run(W)
        val_b = sess.run(b)
        print(step, val_c, val_w, val_b)

        costs.append(val_c)
        weights.append(val_w)
        bs.append(val_b)

print("Learning finished!")

plot.plot(costs, 'o-')
plot.xlabel('Step')
plot.ylabel('Error')
plot.show()

plot.plot(weights, 'o-')
plot.xlabel('Step')
plot.ylabel('Weight')
plot.show()

plot.plot(bs, 'o-')
plot.xlabel('Step')
plot.ylabel('Bias')
plot.show()


