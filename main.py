"""
    Author : Byunghyun Ban
    needleworm@kaist.ac.kr
    latest modification :
        2017.05.01.
"""

import tensorflow as tf
import numpy as np
<<<<<<< Updated upstream
import os
import re
import utils
=======
import re
#import Utils as utils
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import Datareader as dr
import sys
import os
import getopt
#import Evaluator as ev
import GAN_model as GM
>>>>>>> Stashed changes



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


def DCNN(color):
    image = tf.placeholder(tf.float32, [None, FLAGS.YDIM, FLAGS.XDIM])

    layers = (
        ###### Deconvolution Sub-Network ######
        # Relu is not applied on the original paper.
        'index:1_1, type:conv, xsize:121, ysize:1,   stride:1, fm: 1  -> 38'
        #'index:1_1, type:relu' # uncomment this line apply relu
        'index:1_2, type:conv, xsize:1,   ysize:121, stride:1, fm: 38 -> 38'
        #'index:1_2, type:relu' # uncomment this line apply relu
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
            kernels = utils.weight_variable(conv_shape, name=kind+index+"_W"+color)
            bias = utils.bias_variable([conv_shape[-1]], name=kind+index+"_b"+color)
            current = utils.conv2d(current, kernels, bias, stride=stride)
        elif kind=='relu':
            current = tf.nn.relu(current, name=kind+index)
        elif kind == 'deconv':
            deconv_shape, stride = utils.get_conv_shape(name)
            kernels = utils.weight_variable(deconv_shape, name=kind+index+"_dW"+color)
            bias = utils.bias_variable([deconv_shape[-1]], name=kind+index+"_db"+color)
            current = utils.deconv(current, kernels, bias, output_shape=tf.shape(current))
        elif kind == "deconv_out":
            shape = image.get_shape().as_list()
            deconv_shape, stride = utils.get_conv_shape(name)
            kernels = utils.weight_variable(deconv_shape, name=kind+index+"_dW"+color)
            bias = utils.bias_variable([deconv_shape[-1]], name=kind + index + "_db"+color)
            current = utils.deconv(current, kernels, bias, output_shape=(shape[0], FLAGS.YDIM, FLAGS.XDIM, 1),
                                   stride=int(FLAGS.YDIM / current.get_shape().as_list()[1]))
        net[kind+index] = current

    return image, current


class Model(object):
    def __init__(self, batch_size, is_training=True):
        self.low_resolution_image = tf.placeholder(tf.float32, shape=[batch_size,
                                                       FLAGS.YDIM, FLAGS.XDIM], name="input_image")
        self.high_resolution_image = tf.placeholder(tf.float32, shape=[batch_size,
                                                       FLAGS.YDIM, FLAGS.XDIM], name="GT_image")


        trainable_var = tf.trainable_variables()

        self.train_op = train(self.loss, trainable_var)


def train(is_training=True):
    with tf.device(FLAGS.device):
        ###############################  GRAPH PART  ###############################
        print("Graph Initialization...")
        input_R, output_R = DCNN(color="R")
        input_G, output_G = DCNN(color="G")
        input_B, output_B = DCNN(color="B")
        print("Done")

        ############################  Placeholder Part  ############################
        print("Setting up Placeholders...")
        high_resolution_image = tf.placeholder(tf.float32, [None, FLAGS.YDIM, FLAGS.XDIM])

        Loss_R = tf.reduce_sum(tf.square((high_resolution_image[0] - output_R) ** 2) / FLAGS.YDIM / FLAGS.XDIM)
        Loss_G = tf.reduce_sum(tf.square((high_resolution_image[1] - output_G) ** 2) / FLAGS.YDIM / FLAGS.XDIM)
        Loss_B = tf.reduce_sum(tf.square((high_resolution_image[2] - output_B) ** 2) / FLAGS.YDIM / FLAGS.XDIM)
        trainable_var = tf.trainable_variables()
        optimizer_R = tf.train.AdamOptimizer(learning_rate)
        optimizer_G = tf.train.AdamOptimizer(learning_rate)
        optimizer_B = tf.train.AdamOptimizer(learning_rate)

        gradient_R = optimizer_R.compute_gradients(Loss_R, trainable_var)
        gradient_G = optimizer_G.compute_gradients(Loss_G, trainable_var)
        gradient_B = optimizer_B.compute_gradients(Loss_B, trainable_var)

        train_op_R = optimizer_R.appy_gradients(gradient_R)
        train_op_G = optimizer_G.appy_gradients(gradient_G)
        train_op_B = optimizer_B.appy_gradients(gradient_B)
        print("Done")

        ##############################  Summary Part  ##############################
        print("Setting up summary op...")
        loss_placeholder = tf.placeholder(dtype=tf.float32)
        loss_summary_op = tf.summary.scalar("LOSS", loss_placeholder)
        loss_summary_writer = tf.summary.FileWriter(logs_dir + "/loss/")
        mse_R_placeholder = tf.placeholder(dtype=tf.float32)
        mes_Rsummary = tf.summary.scalar("MSE_R", mse_placeholdere)
        mse_Rsummary_writer = tf.summary.FileWriter(logs_dir + "/mse_r/")
        mse_G_placeholder = tf.placeholder(dtype=tf.float32)
        mes_Gsummary = tf.summary.scalar("MSE_G", mse_placeholdere)
        mse_Gsummary_writer = tf.summary.FileWriter(logs_dir + "/mse_g/")
        mse_B_placeholder = tf.placeholder(dtype=tf.float32)
        mes_Bsummary = tf.summary.scalar("MSE_B", mse_placeholdere)
        mse_Bsummary_writer = tf.summary.FileWriter(logs_dir + "/mse_b/")
        print("Done")

        ############################  Model Save Part  #############################
        print("Setting up Saver...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        print("Done")


    ################################  Session Part  ################################
    print("Session Initialization...")
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=sess_config)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("model restored...")
    else:
        sess.run(tf.global_variables_initializer())



def main():
    pass
