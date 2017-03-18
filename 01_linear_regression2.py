from __future__ import print_function

import tensorflow as tf

from regression import Regression
from tfvariable import TFVariable #!!


class LinearRegression2(Regression):
    W = TFVariable.get_var_list(1)
    b = TFVariable.get_var_list(1)

    hypothesis = tf.add(tf.mul(Regression.X, W), b)  # W * x_data + b
    cost_func = tf.reduce_mean(tf.square(hypothesis - Regression.Y))
    l_rate = tf.Variable(0.1)
    optimizer = tf.train.GradientDescentOptimizer(l_rate)
    learning = optimizer.minimize(cost_func)

    sess = tf.Session()
    init = tf.global_variables_initializer()

    def predict(self, x_data):
        return self.sess.run(self.hypothesis, feed_dict={Regression.X: x_data})

    def learn(self, x_data, y_data):

        self.sess.run(self.init)

        print(self.sess.run(self.W))
        self.weights.append(self.sess.run(self.W))
        self.biases.append(self.sess.run(self.b))

        for step in range(2001):
            self.sess.run(self.learning, feed_dict={self.X: x_data, self.Y: y_data})

            if step % 50 == 0:
                val_error = self.sess.run(self.cost_func, feed_dict={self.X: x_data, self.Y: y_data})
                val_weight = self.sess.run(self.W)
                val_bias = self.sess.run(self.b)

                #print(step, 'cost:', val_cost, 'weight:', val_weight, 'bias:', val_bias)

                str = '{} cost:{} weight:{} bias:{}'.format(step, val_error, val_weight, val_bias)
                self.logs.append(str)
                self.costs.append(val_error)
                self.weights.append(val_weight)
                self.biases.append(val_bias)

        print("Learned!")

        print(7.5 * self.sess.run(self.W) + self.sess.run(self.b))


gildong = LinearRegression2()

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

gildong.learn(x_data, y_data)
print(gildong.predict([104.02]))
