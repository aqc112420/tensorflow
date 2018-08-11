import tensorflow as tf
import numpy as np

def generate_data():
    num = 25
    label = np.asarray(range(0,num))
    images = np.random.random([num,5,5,3])
    print('label size:{},image size:{}'.format(label.shape,images.shape))
    return label,images

def get_batch_data():
    label,images = generate_data()
    images = tf.cast(images,tf.float32)
    label = tf.cast(label,tf.float32)
    input_queue = tf.train.slice_input_producer([images,label],shuffle=True,num_epochs=2)
    image_batch,label_batch = tf.train.batch(input_queue,batch_size=6,num_threads=1,capacity=4,allow_smaller_final_batch=False)
    return image_batch,label_batch

image_batch,label_batch = get_batch_data()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord)
    try:
        while not coord.should_stop():
            image_batch_v,label_batch_v = sess.run([image_batch,label_batch])
            print(image_batch_v)
            print(label_batch_v)
            print((image_batch_v.shape))

    except tf.errors.OutOfRangeError:
        print('done')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
