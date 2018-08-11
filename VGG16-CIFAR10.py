import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle as p

batch_size = 128
learning_rate = 0.001
n_iterations = 100
keep_prob = 0.7

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
# encoder = OneHotEncoder()
# Y_train_oh = np.array(encoder.fit_transform(Y_train.reshape(-1,1)).toarray()) #进行标签的one-hot处理
# Y_test_oh = np.array(encoder.fit_transform(Y_test.reshape(-1,1)).toarray()) #进行标签的one-hot处理
X_train,X_test = X_train/255,X_test/255

train_range = list(zip(range(0, len(X_train), batch_size), range(batch_size, len(X_train), batch_size)))

if len(X_train) % batch_size > 0:
    train_range.append((train_range[-1][1], len(X_train)))

# X_train = tf.cast(X_train,tf.float32)
# X_test = tf.cast(X_test,tf.float32)


def VGG16(X_input,keep_prop):
    X_input = tf.reshape(X_input,[-1,32,32,3])
    conv1 = tf.layers.conv2d(inputs=X_input,filters=64,kernel_size=[2,2],padding='same',activation=tf.nn.relu)#32*32*64
    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2)#16*16*64

    conv2 = tf.layers.conv2d(inputs=pool1,filters=128,kernel_size=[2,2],padding='same',activation=tf.nn.relu)#16*16*128
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)#8*8*128

    conv3 = tf.layers.conv2d(inputs=pool2,filters=256,kernel_size=[2,2],padding='same',activation=tf.nn.relu)#8*8*256
    pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=2,strides=2)#4*4*256

    conv4 = tf.layers.conv2d(inputs=pool3,filters=512,kernel_size=[2,2],padding='same',activation=tf.nn.relu)#4*4*512
    pool4 = tf.layers.max_pooling2d(inputs=conv4,pool_size=2,strides=2)#2*2*512

    pool4 = tf.reshape(pool4,[-1,2*2*512])
    fc1 = tf.layers.dense(inputs=pool4,units=512,activation=tf.nn.relu)
    dropout = tf.layers.dropout(fc1,rate=0.3,training=True)

    out = tf.layers.dense(dropout,units=10)
    return out

def compute_loss(logits,labels):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.to_int32(labels)))
    return loss

def compute_accuracy(logits,labels):
    total = tf.equal(tf.cast(tf.argmax(logits,1),tf.float32),labels)
    acc = tf.reduce_sum(tf.reduce_mean(tf.cast(total,tf.float32)))
    return acc

X = tf.placeholder('float',shape=[None,32,32,3],name='x')
Y = tf.placeholder('float',shape=[None,],name='y')

logits = VGG16(X,keep_prop=keep_prob)

with tf.name_scope('loss'):
    loss = compute_loss(logits,Y)
tf.summary.scalar('loss',loss)

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(loss)

with tf.name_scope('accuracy'):
    accuracy = compute_accuracy(logits=logits,labels=Y)
tf.summary.scalar('accuracy',accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(logdir='./log2',graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_iterations):
        for start,end in train_range:
            X_input1,Y_input1 = X_train[start:end],Y_train[start:end]
            _,los,acc = sess.run([optimizer,loss,accuracy],feed_dict={X:X_input1,Y:Y_input1})

            if end %(batch_size * 10) == 0:
                print('Epoch:{}--batch:{} Loss:{},accuracy:{}'.format(epoch,end,los,acc))

        summary = sess.run(merged,feed_dict={X:X_input1,Y:Y_input1})
        train_writer.add_summary(summary,epoch)








