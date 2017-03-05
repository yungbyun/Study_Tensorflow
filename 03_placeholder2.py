from __future__ import print_function
import tensorflow as tf


class XXX:

    X = [1] #tf.placeholder(tf.float32)
    Y = [1] #tf.placeholder(tf.float32)

    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

    hypothesis = W * X + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    a = tf.Variable(0.1)
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

    init = tf.global_variables_initializer()

    def run(self):
        sess = tf.Session()
        sess.run(self.init)
        sess.run(self.train) #, feed_dict={self.X: 1, self.Y: 1})
        print('Learned!')

    def set_data(self, i, j):
        self.a = i
        self.b = j

    def my_func(self):
        add = tf.add(self.a, self.b)

        # Launch the default graph
        sess = tf.Session()
        print(sess.run(add)) #, feed_dict={a: 2, b: 3}))


gildong = XXX()
gildong.set_data(3,4)
gildong.my_func()
gildong.run()