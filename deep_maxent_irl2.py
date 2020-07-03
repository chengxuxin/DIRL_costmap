import numpy as np
import tensorflow as tf
import value_iteration
import os.path
import inference2 as inference
import input_data2 as input_data

'''
used to test lidar img input
use single start point and destination
'''


def int_to_point(W, i):
    return i % W, i // W


def point_to_int(W, x, y):
    return y*W + x


def normalize(vals):
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)


def demo_svf(traj, n_states):
    # compute svf from expert
    p = np.zeros(n_states)

    lenth = len(traj)

    for step in range(lenth):
        p[traj[step]] += 1
    p = p / lenth
    return p


def field_svf(ref):
    # compute svf from expert
    p = ref / np.sum(ref)
    p = np.reshape(p, [-1, ])
    return p


def compute_state_visitation_freq(shape, traj, policy):
    # compute svf from current policy
    [H, W, N_STATES, N_ACTIONS] = shape
    dx = [0, 0, -1, 1, -1, 1, -1, 1]
    dy = [-1, 1, 0, 0, -1, -1, 1, 1]

    T = len(traj)
    # T = 60 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!!!!!!
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T])

    mu[traj[0], 0] += 1
    mu[:, 0] = mu[:, 0] / N_STATES

    for t in range(1, T):
        mu[:, t] = 0.0
        for s in range(N_STATES):
            x, y = int_to_point(W, s)
            for a in range(N_ACTIONS):
                x1 = x + dx[a]
                y1 = y + dy[a]
                if (x1 not in range(W)) or (y1 not in range(H)):
                    continue
                else:
                    s1 = point_to_int(W, x1, y1)
                    mu[s1, t] += mu[s, t - 1] * policy[s, a]
    p = np.sum(mu, 1)
    p = p / np.sum(p)
    return p


def compute_state_visitation_freq_multiple_starts(shape, traj, start, policy):
    # compute svf from current policy
    '''multiple start points'''
    [H, W, N_STATES, N_ACTIONS] = shape
    dx = [0, 0, -1, 1, -1, 1, -1, 1]
    dy = [-1, 1, 0, 0, -1, -1, 1, 1]

    T = len(traj)
    # T = 60 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!!!!!!
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T])

    for i in range(len(start)):
        mu[start[i], 0] += 1
    mu[:, 0] = mu[:, 0] / N_STATES

    for t in range(1, T):
        mu[:, t] = 0.0
        for s in range(N_STATES):
            x, y = int_to_point(W, s)
            for a in range(N_ACTIONS):
                x1 = x + dx[a]
                y1 = y + dy[a]
                if (x1 not in range(W)) or (y1 not in range(H)):
                    continue
                else:
                    s1 = point_to_int(W, x1, y1)
                    mu[s1, t] += mu[s, t - 1] * policy[s, a]
    p = np.sum(mu, 1)
    p = p / np.sum(p)
    return p


def get_queue(IMG_PATH, interval):
    imgs = os.listdir(IMG_PATH)
    imgs.sort(key=lambda x: x[0:5])
    extracts = []
    for index in range(0, len(imgs), interval):  # set extraction interval
        extracts.append(imgs[index])
    return extracts


def get_queue_random(IMG_PATH, interval):
    imgs = os.listdir(IMG_PATH)
    # mgs.sort(key=lambda x: x[0:5])
    extracts = []
    for index in range(0, len(imgs), interval):  # set extraction interval
        extracts.append(imgs[index])
    return extracts


def deep_maxent_irl():

    # hyper parameters
    H = 100
    W = 100
    N_STATES = H * W
    N_ACTIONS = 8
    SHAPE = [H, W, N_STATES, N_ACTIONS]
    DISCOUNT = 1
    LEARNING_RATE_BASE = 0.01
    DECAY_STEPS = 500
    DECAY_RATE = 1
    GRAPH_SAVE_INTERVAL = 10

    IMG_EXTRACT_INTERVAL = 1
    IMG_PATH = '/home/zhuzeyu/real_datasets/ao/img/train/'
    REF_PATH = '/home/zhuzeyu/real_datasets/ao/nav/train/'
    #IMG_PATH = '/Users/David/Desktop/ao/img/train/'
    #REF_PATH = '/Users/David/Desktop/ao/nav/train//'

    # create model directory
    MODEL_DIR = "model"
    if not tf.gfile.Exists(MODEL_DIR):
        tf.gfile.MakeDirs(MODEL_DIR)

    # placeholders
    input_img = tf.placeholder(tf.float32, [None, H, W, 1], name='input_img')
    grad_r_placeholder = tf.placeholder(tf.float32, [H*W, 1])
    # define reward and loss
    rewards = inference.inference(input_img)
    rewards_flattened = tf.reshape(rewards, [N_STATES, 1])
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    l2_loss = tf.reduce_mean([tf.nn.l2_loss(v) for v in theta])
    l2_loss = l2_loss/100000.0
    loss = tf.multiply(grad_r_placeholder, rewards_flattened)
    loss = tf.reduce_sum(loss, name='loss') + l2_loss
    # define training
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        DECAY_STEPS,
        DECAY_RATE
    )
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(loss, global_step=global_step)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=100)

    # get random data
    extracts = get_queue_random(IMG_PATH, IMG_EXTRACT_INTERVAL)

    #path_orig = IMG_PATH + extracts[134]
    #path_ref = REF_PATH + extracts[134]
    #terminal, start, img, traj, ref = input_data.read_img(SHAPE, path_orig, path_ref)

    with tf.Session() as sess:
        init.run()
        for epoch in range(20):
            for iteration in range(len(extracts)):
                path_orig = IMG_PATH + extracts[iteration]
                path_ref = REF_PATH + extracts[iteration]
                terminal, start, img, traj, ref = input_data.read_img(SHAPE, path_orig, path_ref)
                # get rewards
                r = sess.run(rewards, feed_dict={input_img: img})
                r_np = np.reshape(r, [-1, ])

                # get policy
                '''single or multiple destination'''
                terminal_single = traj[len(traj)-1]
                terminal_single2list = []
                terminal_single2list.append(terminal_single)
                _, policy = value_iteration.value_iteration(SHAPE, r_np, DISCOUNT, terminal_single2list)
                # compute expected svf
                '''single or multiple start point'''
                mu_exp = compute_state_visitation_freq(SHAPE, traj, policy)
                # mu_exp = compute_state_visitation_freq_multiple_starts(SHAPE, traj, start, policy)
                # compute expert svf
                '''field svf or traj svf'''
                mu_D = demo_svf(traj, N_STATES)
                #mu_D = field_svf(ref)

                # compute loss
                grad_r = mu_exp - mu_D
                index = np.sum(np.abs(grad_r))
                grad_r = np.reshape(grad_r, [-1, 1])  # 原来是一维
                # train
                sess.run(train_step, feed_dict={grad_r_placeholder: grad_r, input_img: img})

                lss = sess.run(loss, feed_dict={grad_r_placeholder: grad_r, input_img: img})
                print(index)
                print(lss)
                # print(sess.run(l2_loss))

                # save graph
                if iteration % GRAPH_SAVE_INTERVAL == 0:
                    MODEL_NAME = 'model' + str(epoch) + str(iteration) + '.ckpt'
                    saver.save(sess, os.path.join(MODEL_DIR, MODEL_NAME))


if __name__ == '__main__':
    deep_maxent_irl()
