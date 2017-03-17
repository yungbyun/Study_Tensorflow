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



class Animal():
    def speak(self):
        print("...")

class Cat(Animal):
    def speak(self):
        print("meow")

class Dog(Animal):
    def speak(self):
        print("woof")

my_pets = [Dog(), Cat(), Dog()]

for _pet in my_pets:
     _pet.speak()



