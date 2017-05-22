

import tensorflow as tf

def main(_) :
    # Import data

    # Create the model
    x = tf.placeholder(tf.float32, [None,101568]) # 184*184*3 = 33856*3 = 101568 --> image 모양대로 읽는다면.. 필요x
    # x_image = tf.placeholder(tf.float32, [None, 184, 184, 3])
    W_conv1 = weight_variable([1, 121, 1, 38]) # weight initialization for kernel size : 1*121, input channel : 1, output channel : 38
    b_conv1 = bias_variable([38]) # bias initialization for output channel : 38
    x_image = tf.reshape(x, [-1, 184, 184, 3])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)


def weight_variable(shape) :
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape) :
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def conv2d(x, W) :
    return tf.nn.conv2d(x,W, strides=[1, 1, 1, 1], padding='VALID') # no padding


