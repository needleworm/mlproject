"""
    Author : Byunghyun Ban
    needleworm@kaist.ac.kr
    latest modification :
        2017.05.01.
"""

import tensorflow as tf
import numpy as np
import os
import re
import utils



logs_dir = "logs"
results_dir = "results"
data_dir = "low resolution directory"
val_dir = "high resolution directory"


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("device", "/gpu:0", "device : /cpu:0, /gpu:0, /gpu:1. [Default : /gpu:0]")
tf.flags.DEFINE_bool("Train", "True", "mode : train, test. [Default : train]")
tf.flags.DEFINE_bool("reset", "True", "mode : True or False. [Default : train]")
tf.flags.DEFINE_integer("YDIM", "1024", "input image Y size. [Default : 1024]")
tf.flags.DEFINE_integer("XDIM", "1024", "input image X size. [Default : 1024]")
tf.flags.DEFINE_integer("batch_size", "5", "batch size. [Default : 5]")


if FLAGS.reset:
    os.popen("rm -rf " + logs_dir + " " + results_dir)
    os.popen("mkdir " + logs_dir + " " + results_dir)

learning_rate = 0.0001
MAX_ITERATION = int(900)


def DCNN(image):
    layers = (
        ###### Deconvolution Sub-Network ######
        # Relu is not applied on the original paper.
        'index:1_1, type:conv, xsize:121, ysize:1,   stride:1, fm: 1  -> 38'
        #'index:1_1, type:relu'
        'index:1_2, type:conv, xsize:1,   ysize:121, stride:1, fm: 38 -> 38'
        #'index:1_2, type:relu'
        'index:1_3, type:conv, xsize:16,   ysize:16, stride:1, fm: 38 -> 38'

        #### Outlier Rejection Sub-Network ####
        'index:2_1, type:deconv, xsize:1,   ysize:1, stride:1, fm: 38 -> 512'
        'index:2_2, type:deconv, xsize:8,   ysize:8, stride:1, fm: 512 -> 512'

        #### Image Output Layer ####
        'index:3, type:deconv_out, xsize:8,   ysize:8, stride:1, fm: 512 -> 512'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        spec = re.split(':|, |->', name)
        index = spec[1]
        kind = spec[3]
        if kind == 'conv':
            conv_shape, stride = utils.get_conv_shape(name)
            kernels = utils.weight_variable(conv_shape, name=kind+index+"_W")
            bias = utils.bias_variable([conv_shape[-1]], name=kind+index+"_b")
            current = utils.conv2d(current, kernels, bias, stride=stride)
        elif kind=='relu':
            current = tf.nn.relu(current, name=kind+index)
        elif kind == 'deconv':
            deconv_shape, stride = utils.get_conv_shape(name)
            kernels = utils.weight_variable(deconv_shape, name=kind+index+"_dW")
            bias = utils.bias_variable([deconv_shape[-1]], name=kind+index+"_db")
            current = utils.deconv(current, kernels, bias, output_shape=tf.shape(current))
        elif kind == "deconv_out":
            shape = image.get_shape().as_list()
            deconv_shape, stride = utils.get_conv_shape(name)
            kernels = utils.weight_variable(deconv_shape, name=kind+index+"_dW")
            bias = utils.bias_variable([deconv_shape[-1]], name=kind + index + "_db")
            current = utils.deconv(current, kernels, bias, output_shape=(shape[0], FLAGS.YDIM, FLAGS.XDIM, 1),
                                   stride=int(FLAGS.YDIM / current.get_shape().as_list()[1]))
        net[kind+index] = current

    return current


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradient = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(gradient)


class Model(object):
    def __init__(self, batch_size, is_training=True):
        self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        self.low_resolution_image = tf.placeholder(tf.float32, shape=[batch_size,
                                                       FLAGS.YDIM, FLAGS.XDIM], name="input_image")
        self.high_resolution_image = tf.placeholder(tf.float32, shape=[batch_size,
                                                       FLAGS.YDIM, FLAGS.XDIM], name="GT_image")
        self.output = DCNN(self.low_resolution_image)

        self.loss = "Calculate loss with self.output and self.high_resolution_image"
        trainable_var = tf.trainable_variables()

        self.train_op = train(self.loss, trainable_var)


def main():
    pass
