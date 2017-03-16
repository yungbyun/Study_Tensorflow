x_data = [1,2,3]
y_data = [1,2,3]

for (x, y) in zip(x_data, y_data):
    print('hello')

print('hello')
print('hello')
print('hello')
print('hello')
print('hello')


import tensorflow as tf

a = tf.constant([5],dtype=tf.float32)
b = tf.constant([10],dtype=tf.float32)
c = tf.constant([2],dtype=tf.float32)

d = a * b + c

sess = tf.Session()
result = sess.run(d)
print(result)

