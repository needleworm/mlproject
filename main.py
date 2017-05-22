"""
    Author : Byunghyun Ban
    SBIE @ KAIST
    needleworm@kaist.ac.kr
    latest modification :
        2017.04.15.
"""


from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import Utils as utils
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import DataReader as dr
import sys
import os
import getopt
import Evaluator as ev
import GAN_model as GM

__author__ = 'BHBAN, JTKIM'


logs_dir = "logs"
training_data_dir = "images/train"
validation_data_dir = "images/validation"

np.set_printoptions(suppress=True)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "train", "mode : train/ test/ visualize/ evaluation [default : train]")
tf.flags.DEFINE_string("device", "/gpu:0", "device : /cpu:0, /gpu:0, /gpu:1. [Default : /gpu:0]")
tf.flags.DEFINE_bool("Train", "True", "mode : train, test. [Default : train]")
tf.flags.DEFINE_bool("reset", "True", "mode : True or False. [Default : train]")
tf.flags.DEFINE_integer("tr_batch_size", "5", "batch size for training. [default : 5]")
tf.flags.DEFINE_integer("vis_batch_size", "5", "batch size for visualization. [default : 5]")
tf.flags.DEFINE_integer("val_batch_size", "5", "batch size for validation. [default : 5]")

if FLAGS.mode is 'visualize':
    FLAGS.reset = False

if FLAGS.reset:
    print('** Note : directory was reset! **')
    if 'win32' in sys.platform:
        os.popen('rmdir /s /q ' + logs_dir)
    else:
        os.popen('rm -rf ' + logs_dir + '/*')

    os.popen('mkdir ' + logs_dir)
    os.popen('mkdir ' + logs_dir + '/train')
    os.popen('mkdir ' + logs_dir + '/images')
    os.popen('mkdir ' + logs_dir + '/visualize_result')

learning_rate = 0.0001
MAX_ITERATION = int(30000)
IMAGE_RESIZE = 1.0
IMAGE_SIZE = 1024
GT_RESIZE = 1.0
POS_WEIGHT = 0.1
decay = 0.9
stddev = 0.02
stride = 1
keep_prob = 0.5

class GAN:
    def __init__(self, batch_size, is_training=True):
        self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        self.low_resolution_image = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE, 3], name="row_resolution_image")
        self.high_resolution_image = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE, 3], name="high_resolution_image")

        self.Generator = GM.Generator(batch_size, is_training, IMAGE_SIZE, IMAGE_RESIZE, self.keep_probability)
        self.Discriminator = GM.Discriminator(is_training)

        self.D1 = tf.placeholder(tf.float32, shape=[2], name="D1")
        self.D2 = tf.placeholder(tf.float32, shape=[2], name="D2")

        with tf.variable_scope('G'):
            G, _ = self.Generator.generate(self.image, is_training, self.keep_probability, FLAGS.debug)
            I = self.image[:, :, :, FLAGS.window_size]
            self.rgb_predict = tf.concat([I, G, I], axis=3)

        with tf.variable_scope('D') as scope2:
            D1, _ = self.Discriminator.discriminate(self.ground_truth, is_training, self.keep_probability, FLAGS.debug)
            scope2.reuse_variables()
            D2, _ = self.Discriminator.discriminate(self.rgb_predict, is_training, self.keep_probability, FLAGS.debug)

        self.loss = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))

        trainable_var = tf.trainable_variables()

        self.train_op = self.train(trainable_var)

    def train(self, var_list):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradients(self.loss, var_list=var_list)

        return optimizer.apply_gradients(grads)

def train(is_training=True):
    ###############################  GRAPH PART  ###############################
    print("Graph Initialization...")
    with tf.device(FLAGS.device):
        with tf.variable_scope("model", reuse=None):
            m_train = GM.GAN(FLAGS.tr_batch_size, IMAGE_SIZE, IMAGE_RESIZE, keep_prob, is_training=True)
        with tf.variable_scope("model", reuse=True):
            m_valid = GM.GAN(FLAGS.val_batch_size, IMAGE_SIZE, IMAGE_RESIZE, keep_prob, is_training=False)
        with tf.variable_scope("model", reuse=True):
            m_visual = GM.GAN(FLAGS.vis_batch_size, IMAGE_SIZE, IMAGE_RESIZE, keep_prob, is_training=False)
    print("Done")

    ##############################  Summary Part  ##############################
    print("Setting up summary op...")
    loss_ph = tf.placeholder(dtype=tf.float32)
    loss_summary_op = tf.summary.scalar("LOSS", loss_ph)
    psnr_ph = tf.placeholder(dtype=tf.float32)
    psnr_summary_op = tf.summary.scalar("PSNR", psnr_ph)
    generator_summary_writer = tf.summary.FileWriter(logs_dir + '/generator/', max_queue=2)
    discriminator_summary_writer = tf.summary.FileWriter(logs_dir + '/discriminator/', max_queue=2)
    print("Done")

    ############################  Model Save Part  #############################
    print("Setting up Saver...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    print("Done")

    ################################  Session Part  ################################
    print("Session Initialization...")

    if FLAGS.mode == 'train':
        train_dataset_reader = dr.Dataset(path=training_data_dir,
            input_shape=(IMAGE_SIZE*IMAGE_RESIZE, IMAGE_SIZE*IMAGE_RESIZE),
            gt_shape=(IMAGE_SIZE*IMAGE_RESIZE, IMAGE_SIZE*IMAGE_RESIZE))


    for el in vd_folders:
        validation_dataset_reader = dr.Dataset(path=validation_data_dir,
            input_shape=(IMAGE_SIZE*GT_RESIZE, IMAGE_SIZE*GT_RESIZE),
            gt_shape=(IMAGE_SIZE*GT_RESIZE, IMAGE_SIZE*GT_RESIZE))

    val_size = validation_dataset_reader.size
    assert val_size % FLAGS.val_batch_size is 0, "The validation data set size %d must be divided by" \
                                                 " the validation batch size." % val_size
    print("Done")

    print("Session Initialization...")
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    #sess = tf.InteractiveSession(config=sess_config)
    if ckpt and ckpt.model_checkpoint_path:  # model restore
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        sess.run(tf.global_variables_initializer())  # if the checkpoint doesn't exist, do initialization
    print("Done")

     #############################     Train      ###############################
    if FLAGS.mode == "train":

        prev_train_loss = 0
        prev_val_loss = 0

        for itr in range(MAX_ITERATION):
            train_low_resolution_image, train_high_resolution_image = train_dataset_reader.next_batch(FLAGS.tr_batch_size)
            train_dict = {m_train.low_resolution_image: train_low_resolution_image,
                         m_train.high_resolution_image: train_high_resolution_image,
                         m_train.keep_probability: keep_prob}
            sess.run([m_train.train_op], feed_dict=train_dict)

            if itr % 10 == 0:
                valid_low_resolution_image, valid_high_resolution_image = validation_dataset_reader.next_batch(FLAGS.val_batch_size)
                valid_dict = {m_valid.low_resolution_image: valid_low_resolution_image,
                             m_valid.high_resolution_image: valid_high_resolution_image,
                             m_valid.keep_probability: 1.0}

                train_loss, train_pred = sess.run([m_train.loss, m_train.rgb_predict], feed_dict=train_dict)
                valid_loss, valid_pred = sess.run([m_valid.loss, m_valid.rgb_predict], feed_dict=valid_dict)
                train_summary_str = sess.run(loss_summary_op, feed_dict={loss_ph: train_loss})
                valid_summary_str = sess.run(loss_summary_op, feed_dict={loss_ph: valid_loss})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_summary_writer.add_summary(train_summary_str, itr)
                valid_summary_writer.add_summary(valid_summary_str, itr)

                """
                Implement PSNR calculation and save here
                with train_pred and valid_pred
                """

            if itr % 50 == 0:
                saver.save(sess, logs_dir + "/model.ckpt", itr)

            if itr % 500 == 0:
                visual_low_resolution_image, visual_high_resolution_image = validation_dataset_reader.random_batch(FLAGS.val_batch_size)
                visual_dict = {m_valid.low_resolution_image: visual_low_resolution_image,
                               m_valid.high_resolution_image: visual_high_resolution_image,
                               m_valid.keep_probability: 1.0}
                predict = sess.run(m_valid.rgb_predict, feed_dict=visual_dict)

                for i in range(FLAGS.val_batch_size):
                    """
                    Implement here image saving.
                    Input, Output, Ground Truth to be aligned in one JPG image.
                    """

                print('validation images were saved!')

    ###########################     Visualize     ##############################
    elif FLAGS.mode == "visualize":

        visual_low_resolution_image, visual_high_resolution_image = validation_dataset_reader.random_batch(FLAGS.val_batch_size)
        visual_dict = {m_valid.low_resolution_image: visual_low_resolution_image,
                       m_valid.high_resolution_image: visual_high_resolution_image,
                       m_valid.keep_probability: 1.0}
        predict = sess.run(m_valid.rgb_predict, feed_dict=visual_dict)

        for i in range(FLAGS.val_batch_size):
            """
            Implement here image saving.
            Input, Output, Ground Truth to be aligned in one JPG image.
            """
        """
        implement here result plotting.
        use plt.show() 
        """

def main():
    pass
