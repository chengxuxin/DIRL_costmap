import tensorflow as tf
import matplotlib.pyplot as plt
import img_utils
import numpy as np
import cv2 as cv
import input_data
import value_iteration as vi


def int_to_point(W, i):
    return i % W, i // W

def point_to_int(W, x, y):
    return y*W + x


def read_img(shape, i):
    [H, W, N_STATES, N_ACTIONS] = shape
    sim_ref_path = '/Users/David/Desktop/data/sim/sim5_ref.png'
    sim_orig_path = '/Users/David/Desktop/data/sim/sim5_orig.png'
    t_ref_path = '/Users/David/Desktop/DIRL_simplified/DIRL_DataSets/track_ref/' + str(i) + '.png'
    t_orig_path = '/Users/David/Desktop/DIRL_simplified/DIRL_DataSets/orig_img/' + str(i) + '.png'
    t448_ref_path = '/Users/David/Desktop/data/real/448_ref.png'
    t448_orig_path = '/Users/David/Desktop/data/real/448_orig.png'

    ref_raw_data = tf.gfile.GFile(t_ref_path, 'rb').read()
    orig_raw_data = tf.gfile.GFile(t_orig_path, 'rb').read()

    with tf.Session() as sess:
        orig_data = tf.image.decode_jpeg(orig_raw_data)
        resized_orig = tf.image.crop_to_bounding_box(orig_data, 192, 0, 192, 512)
        resized_orig = tf.image.resize_images(resized_orig, [H, W], method=1)
        resized_orig = np.asarray(resized_orig.eval(), dtype='uint8')

    resized_orig = np.reshape(resized_orig, [1, H, W, 3])
    '''added the terminal of the trajectory as return value'''
    return resized_orig


def read_img2(shape, i):
    [H, W, N_STATES, N_ACTIONS] = shape
    t_ref_path = '/Users/David/Desktop/DIRL_simplified/DIRL_DataSets/track_ref/' + str(i) + '.png'
    t_orig_path = '/Users/David/Desktop/DIRL_simplified/DIRL_DataSets/orig_img/' + str(i) + '.png'

    ref_raw_data = tf.gfile.GFile(t_ref_path, 'rb').read()
    orig_raw_data = tf.gfile.GFile(t_orig_path, 'rb').read()

    with tf.Session() as sess:
        ref_data = tf.image.decode_jpeg(ref_raw_data)
        resized_ref = tf.image.crop_to_bounding_box(ref_data, 384, 0, 384, 1024)
        resized_ref = tf.image.resize_images(resized_ref, [H, W], method=1)
        resized_ref = np.asarray(resized_ref.eval(), dtype='uint8')

        orig_data = tf.image.decode_jpeg(orig_raw_data)
        resized_orig = tf.image.crop_to_bounding_box(orig_data, 192, 0, 192, 512)
        resized_orig = tf.image.resize_images(resized_orig, [H, W], method=1)
        resized_orig = np.asarray(resized_orig.eval(), dtype='uint8')

    lenth = int(H-8)
    traj = []
    traj_temp = np.zeros((lenth, 3), dtype=int)
    traj_demo = np.zeros([H, W])
    for i in range(H - 1, H - lenth - 1, -1):
        for j in range(0, W):
            if resized_ref[i][j][1] == 255:
                break

        for k in range(W-1, -1, -1):
            if resized_ref[i][k][1] == 255:
                break
        traj_temp[H - 1 - i][0] = H - 1 - i
        traj_temp[H - 1 - i][1] = i
        traj_temp[H - 1 - i][2] = int((j + k) / 2)
        traj_demo[i][int((j + k) / 2)] = 1

        traj.append(i*W + (j+k)//2)

    resized_orig = np.reshape(resized_orig, [1, H, W, 3])
    terminal = traj[lenth - 1]
    '''added the terminal of the trajectory as return value'''
    return terminal, resized_orig, traj, resized_ref


def read_img3(shape, i):
    [H, W, N_STATES, N_ACTIONS] = shape

    t_ref_path = '/Users/David/Desktop/DIRL_simplified/DIRL_DataSets/track_ref/' + str(i) + '.png'
    t_orig_path = '/Users/David/Desktop/DIRL_simplified/DIRL_DataSets/orig_img/' + str(i) + '.png'

    ref_raw_data = tf.gfile.GFile(t_ref_path, 'rb').read()
    orig_raw_data = tf.gfile.GFile(t_orig_path, 'rb').read()

    with tf.Session() as sess:
        ref_data = tf.image.decode_jpeg(ref_raw_data)
        resized_ref = tf.image.crop_to_bounding_box(ref_data, 384, 0, 384, 1024)
        resized_ref = tf.image.resize_images(resized_ref, [H, W], method=1)
        resized_ref = np.asarray(resized_ref.eval(), dtype='uint8')

        orig_data = tf.image.decode_jpeg(orig_raw_data)
        resized_orig = tf.image.crop_to_bounding_box(orig_data, 192, 0, 192, 512)
        resized_orig = tf.image.resize_images(resized_orig, [H, W], method=1)
        resized_orig = np.asarray(resized_orig.eval(), dtype='uint8')

    traj = []
    terminal = []
    start = []

    for n in range(W):
        if resized_ref[H-1][n][1] == 255:
            start.append(W*(H-1) + n)

    for i in range(H - 1, 0, -1):
        for j in range(0, W):
            if resized_ref[i][j][1] == 255:
                break

        for k in range(W-1, -1, -1):
            if resized_ref[i][k][1] == 255:
                break
        if j == W-1 and k == 0:
            for m in range(W):
                if resized_ref[i+1][m][1] == 255:
                    terminal.append(i*W + m)
            break
        traj.append(i*W + (j+k)//2)

    resized_orig = np.reshape(resized_orig, [1, H, W, 3])
    # terminal = traj[len(traj) - 1]
    '''added the terminal of the trajectory as return value'''
    return terminal, start, resized_orig, traj, resized_ref[:, :, 1]


def compute_state_visitation_freq(shape, traj, policy):
    # compute svf from current policy
    [H, W, N_STATES, N_ACTIONS] = shape
    dx = [0, 0, -1, 1, -1, 1, -1, 1]
    dy = [-1, 1, 0, 0, -1, -1, 1, 1]

    T = len(traj)
    # T = 90  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!!!!!!
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
        # mu[:, t] = mu[:, t] / np.sum(mu[:, t])
    p = np.sum(mu, 1)
    p = p / np.sum(p)
    return p


def compute_state_visitation_freq2(shape, traj, start, policy):
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


H = 48
W = 128
N_STATES = H*W
N_ACTIONS = 8
discount = 1.0
shape = [H, W, N_STATES, N_ACTIONS]

input_img = tf.placeholder(tf.float32, [None, H, W, 3])
sess = tf.Session()
# load meta graph and restore weights
saver = tf.train.import_meta_graph('/Users/David/Desktop/DIRL_simplified/model/model499.ckpt.meta')
saver.restore(sess, '/Users/David/Desktop/DIRL_simplified/model/model499.ckpt')

# get all tensors. not necessary
graph = tf.get_default_graph()
# tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# print(tensor_name_list)

inimg = graph.get_tensor_by_name("input_img:0")
op_to_restore = graph.get_tensor_by_name("rewards:0")

for i in range(1):
    # img = read_img(shape, i+1)
    path_ref = '/Users/David/Desktop/DIRL_simplified/DIRL_DataSets/track_ref/' + str(i+447) + '.png'
    path_orig = '/Users/David/Desktop/DIRL_simplified/DIRL_DataSets/orig_img/' + str(i+447) + '.png'
    terminal, start, img, traj, ref = input_data.read_img(shape, path_orig, path_ref)

    #x, y = int_to_point(W, terminal)
    #print(x, y)

    feed_dict = {inimg: img}

    r = sess.run(op_to_restore, feed_dict)

    # get trajectory from trained reward map
    r = np.reshape(r, [N_STATES, ])

    print(r)
    # binarization
    r_min = np.min(r)
    threshold = 0.35 * r_min
    ret, binarized = cv.threshold(r, threshold, 255, cv.THRESH_BINARY)

    v = 1

    if v == 1:
        # gridworld
        value, policy = vi.value_iteration(shape, r, discount, terminal)
        # value = np.exp(value)
        mu_exp = compute_state_visitation_freq2(shape, traj, start, policy)

        plt.subplot(4, 1, 1)
        # plt.figure(figsize=(H, W))
        img_utils.heatmap2d(np.reshape(mu_exp, (H, W)), 'SVF', block=False, text=False)

        plt.subplot(4, 1, 2)
        # plt.figure(figsize=(H, W))
        img_utils.heatmap2d(np.reshape(value, (H, W)), 'Value', block=False, text=False)

    plt.subplot(4, 1, 3)
    # plt.figure(figsize=(H, W))
    img_utils.heatmap2d(np.reshape(r, (H, W)), 'Reward', block=False, text=False)

    plt.subplot(4, 1, 4)
    # plt.figure(figsize=(H, W))
    img_utils.heatmap2d(np.reshape(binarized, (H, W)), 'Binarized Reward', block=False, text=False)

    plt.show()
    '''
    plt.ion()
    plt.pause(0.1)
    plt.close()
    '''

