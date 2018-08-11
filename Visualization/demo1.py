import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

n_iterations = 1000
lr = 0.001
dropout = 0.9
log_dir = './log'
data_dir = 'F:/Pycharm/tensorflow/mnist'
mnist = input_data.read_data_sets(data_dir,one_hot=True)

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32,shape=[None,784],name='x_input')
    y_ = tf.placeholder(dtype=tf.float32,shape=[None,10],name='y_input')

with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',image_shaped_input,10)

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(initial)

def biaes_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#我们知道，在训练的过程在参数是不断地在改变和优化的，
# 我们往往想知道每次迭代后参数都做了哪些变化，
# 可以将参数的信息展现在tenorbord上，
# 因此我们专门写一个方法来收录每次的参数信息。
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean((var))
        tf.summary.scalar('mean',mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)


def nn_layer(input_tensor,input_dim,output_dim,layer_name,act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim,output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = biaes_variable([output_dim])
            variable_summaries(biases)

        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor,weights) + biases
            tf.summary.histogram('linear',preactivate)
        activations = act(preactivate,name='activation')
        tf.summary.histogram('activations',activations)
        return activations

hidden1 = nn_layer(x,784,500,'layer1')

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability',keep_prob)
    dropped = tf.nn.dropout(hidden1,keep_prob)


y = nn_layer(dropped,500,10,'layer2',act=tf.identity)

with tf.name_scope('loss'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)

tf.summary.scalar('loss',cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)


with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_sum(tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32)))
tf.summary.scalar('accuracy',accuracy)


sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir+'/train',sess.graph)
test_writer = tf.summary.FileWriter(log_dir+'/test')

sess.run(tf.global_variables_initializer())


def feed_dict(train):
    if train:
        xs,ys = mnist.train.next_batch(128)
        k = dropout
    else:
        xs,ys = mnist.test.images[:1000],mnist.test.labels[:1000]
        k = 1.0
    return {x:xs,y_:ys,keep_prob:k}


#每隔10步，就进行一次merge, 并打印一次测试数据集的准确率，
# 然后将测试数据集的各种summary信息写进日志中。
#每隔100步，记录原信息
#其他每一步时都记录下训练集的summary信息并写到日志中。

for i in range(n_iterations):
    if i % 10 == 0:
        summary,acc = sess.run([merged,accuracy],feed_dict=feed_dict(False))
        test_writer.add_summary(summary,i)
        print('Accuracy at step%s:%s'%(i,acc))
    else:
        if i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step],
                                  feed_dict=feed_dict(True),
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
        else:  # Record a summary
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()





