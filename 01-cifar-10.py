import tensorflow as tf
import numpy as np
import pickle as p
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

batch_size = 128

def load_CIFAR_batch(filename):
    with open(filename,'rb') as f:
        datadict = p.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
        Y = np.array(Y)
        return X,Y

def load_CIFAR10(root):
    xs = []
    ys = []
    for i in range(1,6):
        f = root+'/data_batch_{}'.format(i)
        X,Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)

    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    del X,Y
    Xte,Yte = load_CIFAR_batch(root+'/test_batch')

    return Xtr,Ytr,Xte,Yte




#数据预处理
X_train,Y_train,X_test,Y_test = load_CIFAR10('F:/Pycharm/pytorch1/data/cifar-10-batches-py')
encoder = OneHotEncoder()
Y_train_oh = np.array(encoder.fit_transform(Y_train.reshape(-1,1)).toarray()) #进行标签的one-hot处理
Y_test_oh = np.array(encoder.fit_transform(Y_test.reshape(-1,1)).toarray()) #进行标签的one-hot处理
X_train,X_test = X_train/255,X_test/255

train_range = list(zip(range(0, len(X_train), batch_size), range(batch_size, len(X_train), batch_size)))

if len(X_train) % batch_size > 0:
    train_range.append((train_range[-1][1], len(X_train)))

def CNN(X,trainable):
    # inputlayer = X.reshape[-1,32,32,3]
    X = tf.reshape(X,[-1,32,32,3])
    conv1 = tf.layers.conv2d(inputs=X,filters=32,kernel_size=[2,2],padding='same',activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1,pool_size=[2,2],strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[2,2],padding='same',activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

    x = tf.reshape(pool2,[-1,4096])

    fc1 = tf.layers.dense(inputs=x,units=192,activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=fc1,rate=0.3,training=trainable)

    fc2 = tf.layers.dense(dropout,units=10)

    return fc2


def compute_loss(logits,labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels,1),logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    return loss


def compute_accuracy(predicts,labels):
    accuracy = 0
    accuracy += tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predicts,1),tf.argmax(labels,1)),dtype=tf.float32))
    # print(type(predicts))
    # print(labels.shape)
    return accuracy


X_input = tf.placeholder(shape=[None,32,32,3],dtype='float')
Y_input = tf.placeholder(shape=[None,10],dtype='float')

logits = CNN(X_input,trainable=True)
loss = compute_loss(logits=logits,labels=Y_input)
accuracy = compute_accuracy(predicts=logits,labels=Y_input)

train_op = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(100):
        for start,end in train_range:
            X_input1,Y_input1 = X_train[start:end],Y_train_oh[start:end]

            _,pred,los,acc = sess.run([train_op,logits,loss,accuracy],feed_dict={X_input:X_input1,Y_input:Y_input1})

            if end % (batch_size*10) == 0:
                print('Epoch:{}--batch:{} Loss:{},accuracy:{}'.format(epoch,end,los,acc/X_input1.shape[0]))







