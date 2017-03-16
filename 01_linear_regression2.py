from __future__ import print_function

import numpy
import tensorflow as tf
import tfvariable
import myplot


class XXX:
    costs = []
    weights = []
    biases = []
    logs = []

    def show_error(self):
        mp = myplot.MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.costs)

    def show_weight(self):
        mp = myplot.MyPlot()
        mp.set_labels('Step', 'Weight')
        mp.show_list(self.weights)

    def show_bias(self):
        mp = myplot.MyPlot()
        mp.set_labels('Step', 'Bias')
        mp.show_list(self.biases)

    def print_log(self):
        for item in self.logs:
            print(item)

    def learning(self, x_data, y_data):
        sess = tf.Session()

        #W = TFVariable.get_var_list(1)
        #b = TFVariable.get_var_list(1)
        W = tfvariable.TFVariable.get_var('weight')
        b = tfvariable.TFVariable.get_var('bias')

        # tf Graph Input
        #X = tf.placeholder("float")
        #Y = tf.placeholder("float")

        #hypothesis = W * x_data + b
        y_prime = tf.add(tf.mul(x_data, W), b)

        error = tf.reduce_mean(tf.square(y_prime - y_data))

        l_rate = tf.Variable(0.1)
        optimizer = tf.train.GradientDescentOptimizer(l_rate)
        learning = optimizer.minimize(error)

        init = tf.global_variables_initializer()
        sess.run(init)

        print(sess.run(W))

        for step in range(2001):
            sess.run(learning) #, feed_dict={X: x_data, Y: y_data})

            if step % 40 == 0:
                val_error = sess.run(error) #, feed_dict={X: x_data, Y: y_data})
                val_weight = sess.run(W)
                val_bias = sess.run(b)

                #print(step, 'cost:', val_cost, 'weight:', val_weight, 'bias:', val_bias)

                str = '{} cost:{} weight:{} bias:{}'.format(step, val_error, val_weight, val_bias)
                self.logs.append(str)

                self.costs.append(val_error)
                self.weights.append(val_weight)
                self.biases.append(val_bias)

        print("Learning finished!")


gildong = XXX()
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]
gildong.learning(x_data, y_data)
gildong.show_error()
#gildong.show_weight()
#gildong.show_bias()
gildong.print_log()