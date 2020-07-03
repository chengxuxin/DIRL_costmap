import tensorflow as tf

'''
used to test lidar img input
use single start point and destination
'''


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # mean=-0.01!!!!!??????
    weight = tf.Variable(initial_value=initial)
    return weight


# define conv bias
def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    bias = tf.Variable(initial_value=initial)
    return bias


# define conv operation
def conv_op(in_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'):
    conv_out = tf.nn.conv2d(in_tensor, kernel, strides=strides, padding=padding)
    return conv_out


def inference(input_img):
    # weights & bias
    w1 = [5, 5, 1, 64]
    b1 = [64]
    w2 = [3, 3, 64, 32]
    b2 = [32]
    w3 = [3, 3, 32, 32]
    b3 = [32]
    w4 = [1, 1, 32, 32]
    b4 = [32]
    w5 = [1, 1, 32, 1]
    b5 = [1]

    # 1st layer
    W_conv1 = weight_variable(w1)
    b_conv1 = bias_variable(b1)
    h_conv1 = tf.nn.relu(conv_op(input_img, W_conv1) + b_conv1)
    # 2nd layer
    W_conv2 = weight_variable(w2)
    b_conv2 = bias_variable(b2)
    h_conv2 = tf.nn.relu(conv_op(h_conv1, W_conv2) + b_conv2)
    # 3rd layer
    W_conv3 = weight_variable(w3)
    b_conv3 = bias_variable(b3)
    h_conv3 = tf.nn.relu(conv_op(h_conv2, W_conv3) + b_conv3)
    # 4th layer
    W_conv4 = weight_variable(w4)
    b_conv4 = bias_variable(b4)
    h_conv4 = tf.nn.relu(conv_op(h_conv3, W_conv4) + b_conv4)
    # 5th layer
    W_conv5 = weight_variable(w5)
    b_conv5 = bias_variable(b5)
    rewards = tf.add(conv_op(h_conv4, W_conv5), b_conv5)
    rewards = tf.subtract(rewards, tf.reduce_max(rewards), name='rewards')
    '''
    rewards = tf.divide(rewards, tf.abs(tf.reduce_min(rewards)))
    rewards = 100 * rewards
    将reward置于（-100， 0）'''

    return rewards
