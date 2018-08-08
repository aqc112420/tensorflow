import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


tf.set_random_seed(1)
np.random.seed(1)

x = np.linspace(-1,1,100)[:,np.newaxis]
noise = np.random.normal(0,0.1,size=x.shape)
y = np.power(x,2) + noise

# plt.scatter(x,y)
# plt.show()

tf_x = tf.placeholder('float',shape=x.shape)
tf_y = tf.placeholder('float',shape=y.shape)

w1 = tf.Variable(tf.random_normal([1,5]))
b1 = tf.Variable(tf.zeros([1,5])+0.1)
w_plus_b1 = tf.matmul(tf_x,w1) + b1
out1 = tf.nn.relu(w_plus_b1)

w2 = tf.Variable(tf.random_normal([5,10]))
b2 = tf.Variable(tf.zeros([1,10])+0.1)
w_plus_b2 = tf.matmul(out1,w2) + b2
out2 = tf.nn.relu(w_plus_b2)

w3 = tf.Variable(tf.random_normal([10,1]))
b3 = tf.Variable(tf.zeros([1,1])+0.1)
out = tf.matmul(out2,w3) + b3


loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf_y-out),reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # plt.scatter(x,y)
    plt.ion()

    for i in range(1000):
        _,l,pred = sess.run([optimizer,loss,out],feed_dict={tf_x:x,tf_y:y})

        if i % 50 == 0:
            plt.cla()
            plt.scatter(x,y)
            plt.plot(x,pred,'r-',lw=5)
            plt.text(0.5,0,'Loss:{:.4f}'.format(l),fontdict={'size':20,'color':'red'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()

