import tensorflow as tf
import matplotlib.pyplot as plt
import img_utils
import numpy as np
import cv2 as cv
import input_data2 as input_data
import value_iteration as vi


'''
lidar data test
single destination
single start point
'''

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


H = 100
W = 100
N_STATES = H*W
N_ACTIONS = 8
discount = 1.0
SHAPE = [H, W, N_STATES, N_ACTIONS]

is_plt = 0
is_v = 0
is_dir_test = 0
is_binarized = 0
video_name = 'lidar_binarized'

input_img = tf.placeholder(tf.float32, [None, H, W, 3])
sess = tf.Session()
# load meta graph and restore weights
saver = tf.train.import_meta_graph('/Users/David/Desktop/DIRL_simplified/model/model220.ckpt.meta')
saver.restore(sess, '/Users/David/Desktop/DIRL_simplified/model/model220.ckpt')

# get all tensors. not necessary
graph = tf.get_default_graph()
# tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# print(tensor_name_list)

inimg = graph.get_tensor_by_name("input_img:0")
op_to_restore = graph.get_tensor_by_name("rewards:0")

format = cv.VideoWriter_fourcc(*'MJPG')
videoWrite = cv.VideoWriter('lidar.avi', format, 3.0, (H, W), 1)

for i in range(100):
    if is_dir_test:
        index = '00000' + str(i + 108)
        index = index[-5:]
        path_ref = '/Users/David/Desktop/ao/nav/test/' + index + 'e.png'
        path_orig = '/Users/David/Desktop/ao/img/test/' + index + 'e.png'
    else:
        index = '00000' + str(i + 1)
        index = index[-5:]
        path_ref = '/Users/David/Desktop/ao/nav/train/' + index + 'e.png'
        path_orig = '/Users/David/Desktop/ao/img/train/' + index + 'e.png'

    terminal, start, img, traj, ref = input_data.read_img(SHAPE, path_orig, path_ref)

    feed_dict = {inimg: img}
    r = sess.run(op_to_restore, feed_dict)
    r = np.reshape(r, [N_STATES, ])
    print(r)

    # binarization
    r_min = np.min(r)
    threshold = 0.31 * r_min
    ret, binarized = cv.threshold(r, threshold, 255, cv.THRESH_BINARY)

    # get trajectory svf
    traj_svf = np.zeros([H, W])
    for s in range(len(traj)):
        y, x = int_to_point(W, traj[s])
        traj_svf[x][y] = 1

    if is_plt:
        if is_v:
            # gridworld
            '''single or multiple destination'''
            terminal_single = traj[len(traj)-1]
            terminal_single2list = []
            terminal_single2list.append(terminal_single)
            value, policy = vi.value_iteration(SHAPE, r, discount, terminal)
            # value = np.exp(value)
            '''single or multiple start point'''
            mu_exp = compute_state_visitation_freq(SHAPE, traj, policy)
            #mu_exp = compute_state_visitation_freq_multiple_starts(SHAPE, traj, start, policy)

            plt.subplot(2, 4, 5)
            img_utils.heatmap2d(np.reshape(mu_exp, (H, W)), 'Expected SVF', block=False, text=False)

            plt.subplot(2, 4, 6)
            img_utils.heatmap2d(np.reshape(value, (H, W)), 'Value', block=False, text=False)

            plt.subplot(2, 4, 4)
            alpha = 0.8
            integrated_weighted = alpha * 800 * np.reshape(mu_exp, [H, W]) + (1 - alpha) * np.reshape(r, (H, W))
            img_utils.heatmap2d(integrated_weighted, 'Integrated map', block=False, text=False)

        plt.subplot(2, 4, 1)
        img_utils.heatmap2d(np.reshape(ref, [H, W]), 'Expert SVF', block=False, text=False)

        plt.subplot(2, 4, 2)
        img_utils.heatmap2d(traj_svf, 'Trajectory SVF', block=False, text=False)

        plt.subplot(2, 4, 3)
        plt.imshow(np.reshape(img, [H, W]), cmap=plt.cm.gray)
        plt.title('Original image')
        plt.colorbar()

        plt.subplot(2, 4, 7)
        # plt.figure(figsize=(H, W))
        img_utils.heatmap2d(np.reshape(r, (H, W)), 'Reward', block=False, text=False)

        plt.subplot(2, 4, 8)
        # plt.figure(figsize=(H, W))
        img_utils.heatmap2d(np.reshape(binarized, (H, W)), 'Binarized Reward', block=False, text=False)

        plt.show()
        #plt.ion()
        #plt.pause(1)
        #plt.close()

    else:
        if not is_binarized:
            r_reshaped = np.reshape(r, [H, W])
            maxx = np.max(r_reshaped)
            minn = np.min(r_reshaped)
            r_reshaped = r_reshaped - minn
            r_reshaped = 255 * r_reshaped / (maxx - minn)
            stacked = r_reshaped
        else:
            binarized_reshaped = np.reshape(binarized, [H, W])
            maxx = np.max(binarized_reshaped)
            minn = np.min(binarized_reshaped)
            binarized_reshaped = binarized_reshaped - minn
            binarized_reshaped = 255 * binarized_reshaped / (maxx - minn)
            stacked = binarized_reshaped
        #stacked = binarized_reshaped
        #stacked = np.hstack([r_reshaped, binarized_reshaped])
        frame = np.zeros([H, W, 3])
        frame[:, :, 0] = stacked
        frame[:, :, 1] = stacked
        frame[:, :, 2] = stacked

        frame = frame.astype('uint8')
        videoWrite.write(frame)



