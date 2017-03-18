from __future__ import print_function

import tensorflow as tf
from matplotlib import pyplot as plt

from tfvariable import TFVariable
# Graph Input
# Graph Input
# Graph Input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = len(X)

# model weight
W = tf.placeholder(tf.float32)

# Construct a linear model
hypothesis = tf.multiply(X, W)

# Cost function
cost_func = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / m

init = tf.global_variables_initializer()

# for graphs
weights = []
errors = []

# Launch the graphs
sess = tf.Session()
sess.run(init)

print(sess.run(hypothesis, feed_dict={W: 2}))

for i in range(-30, 50):
    #print(i * -0.1, sess.run(cost_func, feed_dict={W: i * 0.1}))
    weights.append(i * 0.1)
    errors.append(sess.run(cost_func, feed_dict={W: i * 0.1}))

plt.plot(weights, errors, 'ro')
plt.ylabel('cost')
plt.xlabel('W')
plt.show()

'''
place holder
우리가 원하는 특정값을 넣어 계산할 수 있는 방법이다?
학습을 다 시킨 후 특정 x값을 주고 결과는 어떻게 되냐?라고 물을 때 어떻게 x값을 줄것인가?
결국 W를 placeholder로 선언했다라는 얘기는? 나중에 특정 w값을 주고 계산하겠다라는 것을 의미.
'''