"""
    Author : Byunghyun Ban
    SBIE @ KAIST
    needleworm@kaist.ac.kr
    latest modification :
        2017.05.23.
"""


from __future__ import print_function
import tensorflow as tf
import numpy as np
import datetime
import Datareader as dr
import Datareader2 as dr2
import sys
import os
import Evaluator as ev
import GAN_model as GM
import utils
import matplotlib.pyplot as plt


__author__ = 'BHBAN'


logs_dir = "logs"
training_data_dir = "images/train/"
validation_data_dir = "images/validation/"

np.set_printoptions(suppress=True)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "train", "mode : train/ test/ visualize/ evaluation [default : train]")
tf.flags.DEFINE_string("device", "/gpu:0", "device : /cpu:0, /gpu:0, /gpu:1. [Default : /gpu:0]")
tf.flags.DEFINE_bool("Train", "True", "mode : train, test. [Default : train]")
tf.flags.DEFINE_bool("reset", "True", "mode : True or False. [Default : train]")
tf.flags.DEFINE_integer("tr_batch_size", "1", "batch size for training. [default : 5]")
tf.flags.DEFINE_integer("vis_batch_size", "1", "batch size for visualization. [default : 5]")
tf.flags.DEFINE_integer("val_batch_size", "1", "batch size for validation. [default : 5]")

if FLAGS.mode is 'visualize':
    FLAGS.reset = False

if FLAGS.reset:
    print('** Note : directory was reset! **')
    if 'win32' in sys.platform:
        os.popen('rmdir /s /q ' + logs_dir)
    else:
        os.popen('rm -rf ' + logs_dir + "/*")
        os.popen('rm -rf ' + logs_dir)

    os.popen('mkdir ' + logs_dir)
    os.popen('mkdir ' + logs_dir + '/train')
    os.popen('mkdir ' + logs_dir + '/valid')
    os.popen('mkdir ' + logs_dir + '/train/psnr')
    os.popen('mkdir ' + logs_dir + '/train/loss_g')
    os.popen('mkdir ' + logs_dir + '/train/loss_d')
    os.popen('mkdir ' + logs_dir + '/valid/psnr')
    os.popen('mkdir ' + logs_dir + '/valid/loss_g')
    os.popen('mkdir ' + logs_dir + '/valid/loss_d')
    os.popen('mkdir ' + logs_dir + '/images')
    os.popen('mkdir ' + logs_dir + '/visualize_result')

learning_rate = 0.0001
MAX_ITERATION = int(30000)
IMAGE_RESIZE = 0.5
IMAGE_SIZE = 1024
GT_RESIZE = 0.5
POS_WEIGHT = 0.1
decay = 0.9
stddev = 0.02
stride = 1
keep_prob = 0.5


class GAN:
    def __init__(self, batch_size, is_training=True):
        self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        self.low_resolution_image = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_SIZE * IMAGE_RESIZE, IMAGE_SIZE * IMAGE_RESIZE, 3], name="low_resolution_image")
        self.high_resolution_image = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_SIZE * GT_RESIZE, IMAGE_SIZE * GT_RESIZE, 3], name="high_resolution_image")

        self.Generator = GM.Generator(batch_size, is_training, IMAGE_SIZE, IMAGE_RESIZE, self.keep_probability)
        self.Discriminator = GM.Discriminator(is_training)

        with tf.variable_scope('G'):
            self.rgb_predict, _ = self.Generator.generate(self.low_resolution_image, is_training, self.keep_probability, IMAGE_SIZE, IMAGE_RESIZE)
        with tf.variable_scope('D') as scope2:
            self.D1, _ = self.Discriminator.discriminate(self.high_resolution_image, is_training, self.keep_probability)
            scope2.reuse_variables()
            self.D2, _ = self.Discriminator.discriminate(self.rgb_predict, is_training, self.keep_probability)

        # basic loss
        #self.loss = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))

        # Goodfellow Loss at NIPS 2016 tutorial
        #self.loss = tf.nn.sigmoid_cross_entropy_with_logits(self.D1, .9) + tf.nn.sigmoid_cross_entropy_with_logits(self.D2, 0.)

        # BEGAN loss
        """
        This model is to generate realistic data.
        As original data to be multiplied is high-quality so we need to modify this.
        by loss_g, Generator is trained to generate high-resolution image.
            Actually it's enough. We don't need to construct GAN when using this loss.
        by loss_d, Discriminator trains to figure out whether given image is real or not.
            With this process, We hope the generator to draw better image.
        """
        self.loss_g = tf.reduce_mean(tf.square(self.rgb_predict - self.high_resolution_image))
        self.loss_d = tf.reduce_mean(tf.log(self.D1) - tf.log(self.D2)) + self.loss_g
        trainable_var = tf.trainable_variables()

        self.train_op_d, self.train_op_g = self.train(trainable_var)

    def train(self, var_list):
        optimizer1 = tf.train.AdamOptimizer(learning_rate)
        optimizer2 = tf.train.AdamOptimizer(learning_rate)
        grads_d = optimizer1.compute_gradients(self.loss_d, var_list=var_list)
        grads_g = optimizer2.compute_gradients(self.loss_g, var_list=var_list)

        return optimizer1.apply_gradients(grads_d), optimizer2.apply_gradients(grads_g)



def train(is_training=True):
    ###############################  GRAPH PART  ###############################
    print("Graph Initialization...")
    with tf.device(FLAGS.device):
        with tf.variable_scope("model", reuse=None):
            m_train = GAN(FLAGS.tr_batch_size, is_training=True)
        with tf.variable_scope("model", reuse=True):
            m_valid = GAN(FLAGS.val_batch_size, is_training=False)
    print("Done")

    ##############################  Summary Part  ##############################
    print("Setting up summary op...")
    loss_d_ph = tf.placeholder(dtype=tf.float32)
    loss_d_summary_op = tf.summary.scalar("LOSS_d", loss_d_ph)
    loss_g_ph = tf.placeholder(dtype=tf.float32)
    loss_g_summary_op = tf.summary.scalar("LOSS_g", loss_g_ph)
    psnr_ph = tf.placeholder(dtype=tf.float32)
    psnr_summary_op = tf.summary.scalar("PSNR", psnr_ph)

    train_summary_writer_d = tf.summary.FileWriter(logs_dir + '/train/loss_d', max_queue=2)
    valid_summary_writer_d = tf.summary.FileWriter(logs_dir + '/valid/loss_d', max_queue=2)
    train_summary_writer_g = tf.summary.FileWriter(logs_dir + '/train/loss_g', max_queue=2)
    valid_summary_writer_g = tf.summary.FileWriter(logs_dir + '/valid/loss_g', max_queue=2)
    train_psnr_writer = tf.summary.FileWriter(logs_dir + '/train/psnr')
    valid_psnr_writer = tf.summary.FileWriter(logs_dir + '/valid/psnr')

    print("Done")

    ############################  Model Save Part  #############################
    print("Setting up Saver...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    print("Done")

    ################################  Session Part  ################################
    print("Session Initialization...")

    validation_dataset_reader = dr2.Dataset(path=validation_data_dir,
                                           input_shape=(int(IMAGE_SIZE*IMAGE_RESIZE), int(IMAGE_SIZE*IMAGE_RESIZE)),
                                           gt_shape=(int(IMAGE_SIZE*GT_RESIZE), int(IMAGE_SIZE*GT_RESIZE)))

    val_size = validation_dataset_reader.max_idx
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
        train_dataset_reader = dr.Dataset(path=training_data_dir,
                                          input_shape=(int(IMAGE_SIZE * IMAGE_RESIZE), int(IMAGE_SIZE * IMAGE_RESIZE)),
                                          gt_shape=(int(IMAGE_SIZE * GT_RESIZE), int(IMAGE_SIZE * GT_RESIZE)))
        for itr in range(MAX_ITERATION):
            train_low_resolution_image, train_high_resolution_image = train_dataset_reader.next_batch(FLAGS.tr_batch_size)
            train_dict = {m_train.low_resolution_image: train_low_resolution_image,
                         m_train.high_resolution_image: train_high_resolution_image,
                         m_train.keep_probability: keep_prob}
            sess.run([m_train.train_op_d, m_train.train_op_g], feed_dict=train_dict)

            if itr % 10 == 0:
                valid_low_resolution_image, valid_high_resolution_image = validation_dataset_reader.next_batch(FLAGS.val_batch_size)
                valid_dict = {m_valid.low_resolution_image: valid_low_resolution_image,
                             m_valid.high_resolution_image: valid_high_resolution_image,
                             m_valid.keep_probability: 1.0}

                train_loss_d, train_loss_g, train_pred = sess.run([m_train.loss_d, m_train.loss_g, m_train.rgb_predict],
                                                                  feed_dict=train_dict)
                valid_loss_d, valid_loss_g, valid_pred = sess.run([m_valid.loss_d, m_valid.loss_g, m_valid.rgb_predict],
                                                                  feed_dict=valid_dict)

                train_summary_str_d, train_summary_str_g = sess.run([loss_d_summary_op, loss_g_summary_op],
                                             feed_dict={loss_d_ph: train_loss_d, loss_g_ph: train_loss_g})
                valid_summary_str_d, valid_summary_str_g = sess.run([loss_d_summary_op, loss_g_summary_op],
                                             feed_dict={loss_d_ph: valid_loss_d, loss_g_ph: valid_loss_g})
                print("%s ---> Validation_loss_discriminator: %g" % (datetime.datetime.now(), valid_loss_d))
                print("%s ---> Validation_loss_generator: %g" % (datetime.datetime.now(), valid_loss_g))
                print("Step: %d, Train_loss_discriminator:%g" % (itr, train_loss_d))
                print("Step: %d, Train_loss_generator:%g" % (itr, train_loss_g))
                train_summary_writer_d.add_summary(train_summary_str_d, itr)
                train_summary_writer_g.add_summary(train_summary_str_g, itr)
                valid_summary_writer_d.add_summary(valid_summary_str_d, itr)
                valid_summary_writer_g.add_summary(valid_summary_str_g, itr)

                print(train_high_resolution_image.dtype)
                print(train_pred.dtype)
                train_high_resolution_image = train_high_resolution_image.astype(np.uint8)
                train_pred = train_pred.astype(np.uint8)
                valid_high_resolution_image = valid_high_resolution_image.astype(np.uint8)
                valid_pred = train_pred.astype(np.uint8)
                train_psnr = ev.psnr(FLAGS.tr_batch_size, train_high_resolution_image, train_pred)
                valid_psnr = ev.psnr(FLAGS.val_batch_size, valid_high_resolution_image, valid_pred)
                train_psnr_str = sess.run(psnr_summary_op, feed_dict={psnr_ph: train_psnr})
                valid_psnr_str = sess.run(psnr_summary_op, feed_dict={psnr_ph: valid_psnr})
                print("%s ---> Validation_PSNR: %g" % (datetime.datetime.now(), valid_psnr))
                print("Step: %d, Train_PSNR:%g" % (itr, train_psnr))
                train_psnr_writer.add_graph(sess.graph)
                valid_psnr_writer.add_graph(sess.graph)
                train_psnr_writer.add_summary(train_psnr_str, itr)
                valid_psnr_writer.add_summary(valid_psnr_str, itr)

            if itr % 50 == 0:
                saver.save(sess, logs_dir + "/model.ckpt", itr)

            if itr % 500 == 0:
                visual_low_resolution_image, visual_high_resolution_image = validation_dataset_reader.random_batch(FLAGS.val_batch_size)
                visual_dict = {m_valid.low_resolution_image: visual_low_resolution_image,
                               m_valid.high_resolution_image: visual_high_resolution_image,
                               m_valid.keep_probability: 1.0}
                predict = sess.run(m_valid.rgb_predict, feed_dict=visual_dict)
                utils.save_images(FLAGS.val_batch_size, logs_dir + '/images', visual_low_resolution_image, predict,
                                  visual_high_resolution_image, show_image=False)
                print('Validation images were saved!')

    ###########################     Visualize     ##############################
    elif FLAGS.mode == "visualize":

        visual_low_resolution_image, visual_high_resolution_image = validation_dataset_reader.random_batch(FLAGS.val_batch_size)
        visual_dict = {m_valid.low_resolution_image: visual_low_resolution_image,
                       m_valid.high_resolution_image: visual_high_resolution_image,
                       m_valid.keep_probability: 1.0}
        predict = sess.run(m_valid.rgb_predict, feed_dict=visual_dict)

        utils.save_images(FLAGS.val_batch_size, validation_data_dir, visual_low_resolution_image, predict,
                                       visual_high_resolution_image, show_image=False)
        print('Validation images were saved!')


def main():
    train(True)
    pass

main()
