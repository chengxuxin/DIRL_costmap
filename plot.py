import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import img_utils

def field_svf(expert_svf, n_states):
    # compute svf from expert
    p = np.zeros(n_states)
    expert_svf = np.reshape(expert_svf, [n_states, 3])
    for i in range(n_states):
        if expert_svf[i][1] == 255:
            p[i] += 1
    p = p / p.sum()
    return p

def read_img(shape):
    [H, W, N_STATES, N_ACTIONS] = shape
    sim_ref_path = '/Users/David/Desktop/data/sim/sim6_ref.png'
    sim_orig_path = '/Users/David/Desktop/data/sim/sim6_orig.png'
    t866_ref_path = '/Users/David/Desktop/data/real/866_ref.png'
    t866_orig_path = '/Users/David/Desktop/data/real/866_orig.png'
    t448_ref_path = '/Users/David/Desktop/data/real/448_ref.png'
    t448_orig_path = '/Users/David/Desktop/data/real/448_orig.png'
    t_ref_path = '/Users/David/Desktop/DIRL_simplified/DIRL_DataSets/track_ref/1.png'
    t_orig_path = '/Users/David/Desktop/DIRL_simplified/DIRL_DataSets/orig_img/1.png'
    t448_ref_path = '/Users/David/Desktop/data/real/448_ref.png'
    t448_orig_path = '/Users/David/Desktop/data/real/448_orig.png'

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

    lenth = int(H-7)
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

    plt.subplot(3, 1, 1)
    plt.imshow(resized_ref)

    plt.subplot(3, 1, 2)
    plt.imshow(traj_demo)

    #plt.subplot(4, 1, 3)
    #plt.imshow(np.reshape(field_svf(resized_ref, N_STATES), [48, 128]), )

    plt.subplot(3, 1, 3)
    plt.imshow(resized_orig)

    plt.show()

    resized_orig = np.reshape(resized_orig, [1, H, W, 3])
    terminal = traj[lenth - 1]
    '''added the terminal of the trajectory as return value'''
    return terminal, resized_orig, traj


if __name__ == '__main__':
    terminal, resized_orig, traj = read_img([48, 128, 48*128, 8])
    print(terminal)