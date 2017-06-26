"""
    Author : Byunghyun Ban
    SBIE @ KAIST
    needleworm@kaist.ac.kr
    latest modification :
        2017.04.15.
"""

__author__ = 'BHBAN'

import tensorflow as tf
import utils as utils

decay=0.9
stddev=0.02

"""
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
"""


class Generator:
    def __init__(self, batch_size, is_training, IMAGE_SIZE, IMAGE_RESIZE, keep_prob=0.5):

        self.Generator_Graph = Generator_Graph(is_training)

    def generate(self, image, is_training, keep_prob, IMAGE_SIZE, IMAGE_RESIZE):
        pred_annotation = self.Generator_Graph.generator(image, is_training, keep_prob, IMAGE_SIZE, IMAGE_RESIZE)

        return pred_annotation


class Generator_Graph:
    def __init__(self, is_training=True):
        self.is_training = is_training
        # Encoder
        self.CNN1_shape  = [1, 121, 3, 38]
        self.CNN1_kernel = tf.get_variable("E_CNN_1_W", initializer=tf.truncated_normal(self.CNN1_shape, stddev=stddev))
        self.CNN1_bias   = tf.get_variable("E_CNN_1_B", initializer=tf.constant(0.0, shape=[self.CNN1_shape[-1]]))

        self.CNN2_shape  = [121, 1, 38, 38]
        self.CNN2_kernel = tf.get_variable("E_CNN_2_W", initializer=tf.truncated_normal(self.CNN2_shape, stddev=stddev))
        self.CNN2_bias   = tf.get_variable("E_CNN_2_B", initializer=tf.constant(0.0, shape=[self.CNN2_shape[-1]]))

        self.CNN3_shape  = [16, 16, 38, 512]
        self.CNN3_kernel = tf.get_variable("E_CNN_3_W", initializer=tf.truncated_normal(self.CNN3_shape, stddev=stddev))
        self.CNN3_bias   = tf.get_variable("E_CNN_3_B", initializer=tf.constant(0.0, shape=[self.CNN3_shape[-1]]))

        self.CNN4_shape  = [1, 1, 512, 512]
        self.CNN4_kernel = tf.get_variable("E_CNN_4_W", initializer=tf.truncated_normal(self.CNN4_shape, stddev=stddev))
        self.CNN4_bias   = tf.get_variable("E_CNN_4_B", initializer=tf.constant(0.0, shape=[self.CNN4_shape[-1]]))

        # Decoder
        self.CNN5_shape  = [8, 8, 512, 3]
        self.CNN5_kernel = tf.get_variable("E_CNN_5_W", initializer=tf.truncated_normal(self.CNN5_shape, stddev=stddev))
        self.CNN5_bias   = tf.get_variable("E_CNN_5_B", initializer=tf.constant(0.0, shape=[self.CNN5_shape[-1]]))

    def _Encoder(self, image, is_training):

    #predict_y = tf.concat([o_conv5_R, o_conv5_G, o_conv5_B], 3)
    #loss = tf.reduce_mean(tf.squared_difference(predict_y, high_resolution_image))
        net = []
        #net.append(image)
        stride=1

        # Conv-Relu-MaxPool 1
        C1 = tf.nn.conv2d(image, self.CNN1_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C1 = tf.contrib.layers.batch_norm(C1, decay=decay, is_training=is_training, updates_collections=None)
        R1 = tf.nn.relu(C1, name="Relu_1")
        #P1 = tf.nn.max_pool(R1, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
        #net.append(R1)

        # Conv-Relu-MaxPool 2
        C2 = tf.nn.conv2d(R1, self.CNN2_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C2 = tf.contrib.layers.batch_norm(C2, decay=decay, is_training=is_training, updates_collections=None)
        R2 = tf.nn.relu(C2, name="Relu_2")
        #P2 = tf.nn.max_pool(R2, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
        #net.append(R2)

        # Conv-Relu-MaxPool 3
        C3 = tf.nn.conv2d(R2, self.CNN3_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C3 = tf.contrib.layers.batch_norm(C3, decay=decay, is_training=is_training, updates_collections=None)
        R3 = tf.nn.relu(C3, name="Relu_3")
        #P3 = tf.nn.max_pool(R3, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
        #net.append(R3)

        # Conv-Relu-MaxPool 4
        C4 = tf.nn.conv2d(R3, self.CNN4_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C4 = tf.contrib.layers.batch_norm(C4, decay=decay, is_training=is_training, updates_collections=None)
        R4 = tf.nn.relu(C4, name="Relu_4")

        C5 = tf.nn.conv2d(R4, self.CNN5_kernel, strides=[1, stride, stride, 1], padding="SAME")
        #C5 = tf.contrib.layers.batch_norm(C4, decay=decay, is_training=is_training, updates_collections=None)
        #R5 = tf.nn.relu(C4, name="Relu_5")


        #P4 = tf.nn.max_pool(R4, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
        net.append(C5)

        return net
        #return C5

    def generator(self, image, is_training, keep_prob, IMAGE_SIZE, IMAGE_RESIZE):
        self.encoder = self._Encoder(image, is_training)
        #self.decoder = self._Decoder(self.encoder, keep_prob, is_training, IMAGE_SIZE, IMAGE_RESIZE)
        #return self.decoder
        return self.encoder

class Discriminator:
    def __init__(self, is_training=True, IMAGE_SIZE=1024, IMAGE_RESIZE=1.0, keep_prob=0.5):

        self.Discriminator_Graph = Discriminator_Graph(is_training)

    def discriminate(self, image, is_training, keep_prob):
        disc, logits = self.Discriminator_Graph.discriminator(image, is_training, keep_prob)
        return disc, logits


class Discriminator_Graph:
    def __init__(self, is_training=True):
        self.is_training = is_training

        # VGG Net
        #  64
        self.CNN1_shape = [2, 2, 3, 64]
        self.CNN1_kernel = tf.get_variable("DISC_CNN_1_W", initializer=tf.truncated_normal(self.CNN1_shape, stddev=stddev))
        self.CNN1_bias = tf.get_variable("DISC_CNN_1_B", initializer=tf.constant(0.0, shape=[self.CNN1_shape[-1]]))

        #  128
        self.CNN3_shape = [2, 2, 64, 128]
        self.CNN3_kernel = tf.get_variable("DISC_CNN_3_W", initializer=tf.truncated_normal(self.CNN3_shape, stddev=stddev))
        self.CNN3_bias = tf.get_variable("DISC_CNN_3_B", initializer=tf.constant(0.0, shape=[self.CNN3_shape[-1]]))

        #  256
        self.CNN5_shape = [2, 2, 128, 256]
        self.CNN5_kernel = tf.get_variable("DISC_CNN_5_W", initializer=tf.truncated_normal(self.CNN5_shape, stddev=stddev))
        self.CNN5_bias = tf.get_variable("DISC_CNN_5_B", initializer=tf.constant(0.0, shape=[self.CNN5_shape[-1]]))

        #  512
        self.CNN9_shape = [2, 2, 256, 512]
        self.CNN9_kernel = tf.get_variable("DISC_CNN_9_W", initializer=tf.truncated_normal(self.CNN9_shape, stddev=stddev))
        self.CNN9_bias = tf.get_variable("DISC_CNN_9_B", initializer=tf.constant(0.0, shape=[self.CNN9_shape[-1]]))

        self.CNN16_shape = [2, 2, 512, 512]
        self.CNN16_kernel = tf.get_variable("DISC_CNN_16_W", initializer=tf.truncated_normal(self.CNN16_shape, stddev=stddev))
        self.CNN16_bias = tf.get_variable("DISC_CNN_16_B", initializer=tf.constant(0.0, shape=[self.CNN16_shape[-1]]))

        # Fully Connected Networks
        self.FNN17_shape = [512, 4096]
        self.FNN17_kernel = tf.get_variable("DISC_FNN_17_W", initializer=tf.truncated_normal(self.FNN17_shape, stddev=stddev))
        self.FNN17_bias = tf.get_variable("DISC_FNN_17_B", initializer=tf.constant(0.0, shape=[self.FNN17_shape[-1]]))

        self.FNN18_shape = [4096, 4096]
        self.FNN18_kernel = tf.get_variable("DISC_FNN_18_W", initializer=tf.truncated_normal(self.FNN18_shape, stddev=stddev))
        self.FNN18_bias = tf.get_variable("DISC_FNN_18_B", initializer=tf.constant(0.0, shape=[self.FNN18_shape[-1]]))

        self.FNN19_shape = [4096, 1000]
        self.FNN19_kernel = tf.get_variable("DISC_FNN_19_W", initializer=tf.truncated_normal(self.FNN19_shape, stddev=stddev))
        self.FNN19_bias = tf.get_variable("DISC_FNN_19_B", initializer=tf.constant(0.0, shape=[self.FNN19_shape[-1]]))

        self.FNN20_shape = [1000, 2]
        self.FNN20_kernel = tf.get_variable("DISC_FNN_20_W", initializer=tf.truncated_normal(self.FNN20_shape, stddev=stddev))
        self.FNN20_bias = tf.get_variable("DISC_FNN_20_B", initializer=tf.constant(0.0, shape=[self.FNN20_shape[-1]]))

    def Graph(self, image, is_training, keep_prob):
        net = []
        net.append(image)
        stride=1
        # Conv-Relu-Conv-Relu-Maxpool
        C1 = tf.nn.conv2d(image, self.CNN1_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C1 = tf.contrib.layers.batch_norm(C1, decay=decay, is_training=is_training, updates_collections=None)
        R1 = tf.nn.relu(C1, name="DISC_Relu_1")
        P1 = tf.nn.max_pool(R1, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
        net.append(P1)

        # Conv-Relu-Conv-Relu-Maxpool
        C3 = tf.nn.conv2d(P1, self.CNN3_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C3 = tf.contrib.layers.batch_norm(C3, decay=decay, is_training=is_training, updates_collections=None)
        R3 = tf.nn.relu(C3, name="DISC_Relu_3")

        P2 = tf.nn.max_pool(R3, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
        net.append(P2)

        # Conv-Relu * 4 times + Maxpool
        C5 = tf.nn.conv2d(P2, self.CNN5_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C5 = tf.contrib.layers.batch_norm(C5, decay=decay, is_training=is_training, updates_collections=None)
        R5 = tf.nn.relu(C5, name="DISC_Relu_5")

        P3 = tf.nn.max_pool(R5, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
        net.append(P3)

        # Conv-Relu * 4 times + Maxpool
        C9 = tf.nn.conv2d(P3, self.CNN9_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C9 = tf.contrib.layers.batch_norm(C9, decay=decay, is_training=is_training, updates_collections=None)
        R9 = tf.nn.relu(C9, name="DISC_Relu_9")
        P4 = tf.nn.max_pool(R9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        net.append(P4)

        # Conv-Relu * 4 times + Maxpool

        C16 = tf.nn.conv2d(P4, self.CNN16_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C16 = tf.contrib.layers.batch_norm(C16, decay=decay, is_training=is_training, updates_collections=None)
        R16 = tf.nn.relu(C16, name="DISC_Relu_16")

        P5 = tf.nn.max_pool(R16, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        net.append(P5)

        # FC-Relu-FC-Relu-FC-SoftMax
        F17 = tf.matmul(tf.reshape(P5, [-1, self.FNN17_shape[0]]), self.FNN17_kernel)
        F17 = tf.nn.bias_add(F17, self.FNN17_bias)
        R17 = tf.nn.relu(F17, name="DISC_Relu_17")

        F18 = tf.matmul(R17, self.FNN18_kernel)
        F18 = tf.nn.bias_add(F18, self.FNN18_bias)
        R18 = tf.nn.relu(F18, name="DISC_Relu_18")

        F19 = tf.matmul(R18, self.FNN19_kernel)
        F19 = tf.nn.bias_add(F19, self.FNN19_bias)
        R19 = tf.nn.relu(F19, name="DISC_Relu_19")

        F20 = tf.matmul(R19, self.FNN20_kernel)
        F20 = tf.nn.bias_add(F20, self.FNN20_bias)

        out = tf.nn.sigmoid(F20)

        net.append(out)

        return out, net

    def discriminator(self, image, is_training, keep_prob):
        return self.Graph(image, is_training, keep_prob)


class Discriminator_Graph_long:
    def __init__(self, is_training=True):
        self.is_training = is_training

        # VGG Net
        #  64
        self.CNN1_shape = [2, 2, 3, 64]
        self.CNN1_kernel = tf.get_variable("DISC_CNN_1_W", initializer=tf.truncated_normal(self.CNN1_shape, stddev=stddev))
        self.CNN1_bias = tf.get_variable("DISC_CNN_1_B", initializer=tf.constant(0.0, shape=[self.CNN1_shape[-1]]))

        self.CNN2_shape = [2, 2, 64, 64]
        self.CNN2_kernel = tf.get_variable("DISC_CNN_2_W", initializer=tf.truncated_normal(self.CNN2_shape, stddev=stddev))
        self.CNN2_bias = tf.get_variable("DISC_CNN_2_B", initializer=tf.constant(0.0, shape=[self.CNN2_shape[-1]]))

        #  128
        self.CNN3_shape = [2, 2, 64, 128]
        self.CNN3_kernel = tf.get_variable("DISC_CNN_3_W", initializer=tf.truncated_normal(self.CNN3_shape, stddev=stddev))
        self.CNN3_bias = tf.get_variable("DISC_CNN_3_B", initializer=tf.constant(0.0, shape=[self.CNN3_shape[-1]]))

        self.CNN4_shape = [2, 2, 128, 128]
        self.CNN4_kernel = tf.get_variable("DISC_CNN_4_W", initializer=tf.truncated_normal(self.CNN4_shape, stddev=stddev))
        self.CNN4_bias = tf.get_variable("DISC_CNN_4_B", initializer=tf.constant(0.0, shape=[self.CNN4_shape[-1]]))

        #  256
        self.CNN5_shape = [2, 2, 128, 256]
        self.CNN5_kernel = tf.get_variable("DISC_CNN_5_W", initializer=tf.truncated_normal(self.CNN5_shape, stddev=stddev))
        self.CNN5_bias = tf.get_variable("DISC_CNN_5_B", initializer=tf.constant(0.0, shape=[self.CNN5_shape[-1]]))

        self.CNN6_shape = [2, 2, 256, 256]
        self.CNN6_kernel = tf.get_variable("DISC_CNN_6_W", initializer=tf.truncated_normal(self.CNN6_shape, stddev=stddev))
        self.CNN6_bias = tf.get_variable("DISC_CNN_6_B", initializer=tf.constant(0.0, shape=[self.CNN6_shape[-1]]))

        self.CNN7_shape = [2, 2, 256, 256]
        self.CNN7_kernel = tf.get_variable("DISC_CNN_7_W", initializer=tf.truncated_normal(self.CNN7_shape, stddev=stddev))
        self.CNN7_bias = tf.get_variable("DISC_CNN_7_B", initializer=tf.constant(0.0, shape=[self.CNN7_shape[-1]]))

        self.CNN8_shape = [2, 2, 256, 256]
        self.CNN8_kernel = tf.get_variable("DISC_CNN_8_W", initializer=tf.truncated_normal(self.CNN8_shape, stddev=stddev))
        self.CNN8_bias = tf.get_variable("DISC_CNN_8_B", initializer=tf.constant(0.0, shape=[self.CNN8_shape[-1]]))

        #  512
        self.CNN9_shape = [2, 2, 256, 512]
        self.CNN9_kernel = tf.get_variable("DISC_CNN_9_W", initializer=tf.truncated_normal(self.CNN9_shape, stddev=stddev))
        self.CNN9_bias = tf.get_variable("DISC_CNN_9_B", initializer=tf.constant(0.0, shape=[self.CNN9_shape[-1]]))

        self.CNN10_shape = [2, 2, 512, 512]
        self.CNN10_kernel = tf.get_variable("DISC_CNN_10_W", initializer=tf.truncated_normal(self.CNN10_shape, stddev=stddev))
        self.CNN10_bias = tf.get_variable("DISC_CNN_10_B", initializer=tf.constant(0.0, shape=[self.CNN10_shape[-1]]))

        self.CNN11_shape = [2, 2, 512, 512]
        self.CNN11_kernel = tf.get_variable("DISC_CNN_11_W", initializer=tf.truncated_normal(self.CNN11_shape, stddev=stddev))
        self.CNN11_bias = tf.get_variable("DISC_CNN_11_B", initializer=tf.constant(0.0, shape=[self.CNN11_shape[-1]]))

        self.CNN12_shape = [2, 2, 512, 512]
        self.CNN12_kernel = tf.get_variable("DISC_CNN_12_W", initializer=tf.truncated_normal(self.CNN12_shape, stddev=stddev))
        self.CNN12_bias = tf.get_variable("DISC_CNN_12_B", initializer=tf.constant(0.0, shape=[self.CNN12_shape[-1]]))

        self.CNN13_shape = [2, 2, 512, 512]
        self.CNN13_kernel = tf.get_variable("DISC_CNN_13_W", initializer=tf.truncated_normal(self.CNN13_shape, stddev=stddev))
        self.CNN13_bias = tf.get_variable("DISC_CNN_13_B", initializer=tf.constant(0.0, shape=[self.CNN13_shape[-1]]))

        self.CNN14_shape = [2, 2, 512, 512]
        self.CNN14_kernel = tf.get_variable("DISC_CNN_14_W", initializer=tf.truncated_normal(self.CNN14_shape, stddev=stddev))
        self.CNN14_bias = tf.get_variable("DISC_CNN_14_B", initializer=tf.constant(0.0, shape=[self.CNN14_shape[-1]]))

        self.CNN15_shape = [2, 2, 512, 512]
        self.CNN15_kernel = tf.get_variable("DISC_CNN_15_W", initializer=tf.truncated_normal(self.CNN15_shape, stddev=stddev))
        self.CNN15_bias = tf.get_variable("DISC_CNN_15_B", initializer=tf.constant(0.0, shape=[self.CNN15_shape[-1]]))

        self.CNN16_shape = [2, 2, 512, 512]
        self.CNN16_kernel = tf.get_variable("DISC_CNN_16_W", initializer=tf.truncated_normal(self.CNN16_shape, stddev=stddev))
        self.CNN16_bias = tf.get_variable("DISC_CNN_16_B", initializer=tf.constant(0.0, shape=[self.CNN16_shape[-1]]))

        # Fully Connected Networks
        self.FNN17_shape = [512, 4096]
        self.FNN17_kernel = tf.get_variable("DISC_FNN_17_W", initializer=tf.truncated_normal(self.FNN17_shape, stddev=stddev))
        self.FNN17_bias = tf.get_variable("DISC_FNN_17_B", initializer=tf.constant(0.0, shape=[self.FNN17_shape[-1]]))

        self.FNN18_shape = [4096, 4096]
        self.FNN18_kernel = tf.get_variable("DISC_FNN_18_W", initializer=tf.truncated_normal(self.FNN18_shape, stddev=stddev))
        self.FNN18_bias = tf.get_variable("DISC_FNN_18_B", initializer=tf.constant(0.0, shape=[self.FNN18_shape[-1]]))

        self.FNN19_shape = [4096, 1000]
        self.FNN19_kernel = tf.get_variable("DISC_FNN_19_W", initializer=tf.truncated_normal(self.FNN19_shape, stddev=stddev))
        self.FNN19_bias = tf.get_variable("DISC_FNN_19_B", initializer=tf.constant(0.0, shape=[self.FNN19_shape[-1]]))

        self.FNN20_shape = [1000, 2]
        self.FNN20_kernel = tf.get_variable("DISC_FNN_20_W", initializer=tf.truncated_normal(self.FNN20_shape, stddev=stddev))
        self.FNN20_bias = tf.get_variable("DISC_FNN_20_B", initializer=tf.constant(0.0, shape=[self.FNN20_shape[-1]]))

    def Graph(self, image, is_training, keep_prob):
        net = []
        net.append(image)
        stride=1
        # Conv-Relu-Conv-Relu-Maxpool
        C1 = tf.nn.conv2d(image, self.CNN1_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C1 = tf.nn.bias_add(C1, self.CNN1_bias)
        C1 = tf.contrib.layers.batch_norm(C1, decay=decay, is_training=is_training, updates_collections=None)
        R1 = tf.nn.relu(C1, name="DISC_Relu_1")

        C2 = tf.nn.conv2d(R1, self.CNN2_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C2 = tf.nn.bias_add(C2, self.CNN2_bias)
        C2 = tf.contrib.layers.batch_norm(C2, decay=decay, is_training=is_training, updates_collections=None)
        R2 = tf.nn.relu(C2, name="DISC_Relu_2")

        P1 = tf.nn.max_pool(R2, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
        net.append(P1)

        # Conv-Relu-Conv-Relu-Maxpool
        C3 = tf.nn.conv2d(P1, self.CNN3_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C3 = tf.nn.bias_add(C3, self.CNN3_bias)
        C3 = tf.contrib.layers.batch_norm(C3, decay=decay, is_training=is_training, updates_collections=None)
        R3 = tf.nn.relu(C3, name="DISC_Relu_3")

        C4 = tf.nn.conv2d(R3, self.CNN4_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C4 = tf.nn.bias_add(C4, self.CNN4_bias)
        C4 = tf.contrib.layers.batch_norm(C4, decay=decay, is_training=is_training, updates_collections=None)
        R4 = tf.nn.relu(C4, name="DISC_Relu_4")

        P2 = tf.nn.max_pool(R4, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
        net.append(P2)

        # Conv-Relu * 4 times + Maxpool
        C5 = tf.nn.conv2d(P2, self.CNN5_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C5 = tf.nn.bias_add(C5, self.CNN5_bias)
        C5 = tf.contrib.layers.batch_norm(C5, decay=decay, is_training=is_training, updates_collections=None)
        R5 = tf.nn.relu(C5, name="DISC_Relu_5")

        C6 = tf.nn.conv2d(R5, self.CNN6_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C6 = tf.nn.bias_add(C6, self.CNN6_bias)
        C6 = tf.contrib.layers.batch_norm(C6, decay=decay, is_training=is_training, updates_collections=None)
        R6 = tf.nn.relu(C6, name="DISC_Relu_6")

        C7 = tf.nn.conv2d(R6, self.CNN7_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C7 = tf.nn.bias_add(C7, self.CNN7_bias)
        C7 = tf.contrib.layers.batch_norm(C7, decay=decay, is_training=is_training, updates_collections=None)
        R7 = tf.nn.relu(C7, name="DISC_Relu_7")

        C8 = tf.nn.conv2d(R7, self.CNN8_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C8 = tf.nn.bias_add(C8, self.CNN8_bias)
        C8 = tf.contrib.layers.batch_norm(C8, decay=decay, is_training=is_training, updates_collections=None)
        R8 = tf.nn.relu(C8, name="DISC_Relu_8")

        P3 = tf.nn.max_pool(R8, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
        net.append(P3)

        # Conv-Relu * 4 times + Maxpool
        C9 = tf.nn.conv2d(P3, self.CNN9_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C9 = tf.nn.bias_add(C9, self.CNN9_bias)
        C9 = tf.contrib.layers.batch_norm(C9, decay=decay, is_training=is_training, updates_collections=None)
        R9 = tf.nn.relu(C9, name="DISC_Relu_9")

        C10 = tf.nn.conv2d(R9, self.CNN10_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C10 = tf.nn.bias_add(C10, self.CNN10_bias)
        C10 = tf.contrib.layers.batch_norm(C10, decay=decay, is_training=is_training, updates_collections=None)
        R10 = tf.nn.relu(C10, name="DISC_Relu_10")

        C11 = tf.nn.conv2d(R10, self.CNN11_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C11 = tf.nn.bias_add(C11, self.CNN11_bias)
        C11 = tf.contrib.layers.batch_norm(C11, decay=decay, is_training=is_training, updates_collections=None)
        R11 = tf.nn.relu(C11, name="DISC_Relu_11")

        C12 = tf.nn.conv2d(R11, self.CNN12_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C12 = tf.nn.bias_add(C12, self.CNN12_bias)
        C12 = tf.contrib.layers.batch_norm(C12, decay=decay, is_training=is_training, updates_collections=None)
        R12 = tf.nn.relu(C12, name="DISC_Relu_12")

        P4 = tf.nn.max_pool(R12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        net.append(P4)

        # Conv-Relu * 4 times + Maxpool
        C13 = tf.nn.conv2d(P4, self.CNN13_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C13 = tf.nn.bias_add(C13, self.CNN13_bias)
        C13 = tf.contrib.layers.batch_norm(C13, decay=decay, is_training=is_training, updates_collections=None)
        R13 = tf.nn.relu(C13, name="DISC_Relu_13")

        C14 = tf.nn.conv2d(R13, self.CNN14_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C14 = tf.nn.bias_add(C14, self.CNN14_bias)
        C14 = tf.contrib.layers.batch_norm(C14, decay=decay, is_training=is_training, updates_collections=None)
        R14 = tf.nn.relu(C14, name="DISC_Relu_14")

        C15 = tf.nn.conv2d(R14, self.CNN15_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C15 = tf.nn.bias_add(C15, self.CNN15_bias)
        C15 = tf.contrib.layers.batch_norm(C15, decay=decay, is_training=is_training, updates_collections=None)
        R15 = tf.nn.relu(C15, name="DISC_Relu_15")

        C16 = tf.nn.conv2d(R15, self.CNN16_kernel, strides=[1, stride, stride, 1], padding="SAME")
        C16 = tf.nn.bias_add(C16, self.CNN16_bias)
        C16 = tf.contrib.layers.batch_norm(C16, decay=decay, is_training=is_training, updates_collections=None)
        R16 = tf.nn.relu(C16, name="DISC_Relu_16")

        P5 = tf.nn.max_pool(R16, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        net.append(P5)

        # FC-Relu-FC-Relu-FC-SoftMax
        F17 = tf.matmul(tf.reshape(P5, [-1, self.FNN17_shape[0]]), self.FNN17_kernel)
        F17 = tf.nn.bias_add(F17, self.FNN17_bias)
        R17 = tf.nn.relu(F17, name="DISC_Relu_17")

        F18 = tf.matmul(R17, self.FNN18_kernel)
        F18 = tf.nn.bias_add(F18, self.FNN18_bias)
        R18 = tf.nn.relu(F18, name="DISC_Relu_18")

        F19 = tf.matmul(R18, self.FNN19_kernel)
        F19 = tf.nn.bias_add(F19, self.FNN19_bias)
        R19 = tf.nn.relu(F19, name="DISC_Relu_19")

        F20 = tf.matmul(R19, self.FNN20_kernel)
        F20 = tf.nn.bias_add(F20, self.FNN20_bias)

        out = tf.nn.softmax(F20)

        net.append(out)

        return out, net

    def discriminator(self, image, is_training, keep_prob):
        return self.Graph(image, is_training, keep_prob)
