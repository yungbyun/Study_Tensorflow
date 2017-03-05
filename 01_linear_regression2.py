from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plot
import numpy
import myplot


class XXX:
    def tf_random_var(self, name):
        #randn 파라미터가 없을 경우 randn 함수는 단일 Python float 값 리턴
        return tf.Variable(numpy.random.randn(), name)

    def tf_random_vars(self):
        return tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #리스트로

    def run(self):
        sess = tf.Session()

        rng = numpy.random

        x_data = [1., 2., 3.]
        y_data = [1., 2., 3.]

        W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #리스트로
        b1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #리스트로

        W = self.tf_random_var('weight')
        b = tf.Variable(rng.randn(), name="bias") #단일값으로

        # tf Graph Input
        #X = tf.placeholder("float")
        #Y = tf.placeholder("float")

        #hypothesis = W * x_data + b
        y_prime = tf.add(tf.mul(x_data, W), b)

        error = tf.reduce_mean(tf.square(y_prime - y_data))

        l_rate = tf.Variable(0.1)
        optimizer = tf.train.GradientDescentOptimizer(l_rate)
        learning = optimizer.minimize(error)

        costs = []
        weights = []
        biases = []
        logs = []

        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(init)

        print(sess.run(W1))

        for step in range(2001):
            sess.run(learning) #, feed_dict={X: x_data, Y: y_data})

            if step % 40 == 0:
                val_cost = sess.run(error) #, feed_dict={X: x_data, Y: y_data})
                val_weight = sess.run(W)
                val_bias = sess.run(b)

                #print(step, 'cost:', val_cost, 'weight:', val_weight, 'bias:', val_bias)

                str = '{} cost:{} weight:{} bias:{}'.format(step, val_cost, val_weight, val_bias)
                logs.append(str)

                costs.append(val_cost)
                weights.append(val_weight)
                biases.append(val_bias)

        print("Learning finished!")

        for item in logs:
            print(item)

        '''

        gildong = myplot.MyPlot()
        gildong.set_labels('Step', 'Cost')
        gildong.show_list(costs)

        gildong.set_labels('Step', 'Weight')
        gildong.show_list(weights)

        gildong.set_labels('Step', 'Bias')
        gildong.show_list(biases)
        '''

gildong = XXX()
gildong.run()

