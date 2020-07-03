import tensorflow as tf
import numpy as np

def read_img(shape, path_orig, path_ref):
    [H, W, N_STATES, N_ACTIONS] = shape

    ref_raw_data = tf.gfile.GFile(path_ref, 'rb').read()
    orig_raw_data = tf.gfile.GFile(path_orig, 'rb').read()

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


if __name__ == '__main__':
    H = 48
    W = 128
    N_STATES = H * W
    N_ACTIONS = 8
    SHAPE = [H, W, N_STATES, N_ACTIONS]
    IMG_PATH = '/Users/David/Desktop/DIRL_simplified/DIRL_DataSets/orig_img/1.png'
    REF_PATH = '/Users/David/Desktop/DIRL_simplified/DIRL_DataSets/track_ref/1.png'
    read_img(SHAPE, IMG_PATH, REF_PATH)