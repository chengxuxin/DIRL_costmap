import cv2 as cv
import tensorflow as tf
import numpy as np


def read_img(shape, path_orig, path_ref):
    [H, W, N_STATES, N_ACTIONS] = shape

    ref_raw_data = cv.imread(path_ref, cv.IMREAD_GRAYSCALE)
    orig_raw_data = cv.imread(path_orig, cv.IMREAD_GRAYSCALE)

    resized_ref = ref_raw_data
    resized_orig = orig_raw_data

    traj = []
    terminal = []
    start = []

    for n in range(W):
        if resized_ref[H-1][n] == 1:
            start.append(W*(H-1) + n)

    for i in range(H - 1, -1, -1):
        for j in range(0, W):
            if resized_ref[i][j] == 1:
                break

        for k in range(W-1, -1, -1):
            if resized_ref[i][k] == 1:
                break
        if j == W-1 and k == 0 or i == 0:
            for m in range(W):
                if resized_ref[i+1][m] == 1:
                    terminal.append(i*W + m)
            if i != 0:
                break
        traj.append(i*W + (j+k)//2)

    resized_orig = np.reshape(resized_orig, [1, H, W, 1])
    # terminal = traj[len(traj) - 1]
    '''added the terminal of the trajectory as return value'''

    return terminal, start, resized_orig, traj, np.reshape(resized_ref, [H, W, 1])



if __name__ == '__main__':
    H = 100
    W = 100
    N_STATES = H * W
    N_ACTIONS = 8
    SHAPE = [H, W, N_STATES, N_ACTIONS]
    IMG_PATH = '/Users/David/Desktop/ao/img/00134e.png'
    REF_PATH = '/Users/David/Desktop/ao/nav/00134e.png'
    read_img(SHAPE, IMG_PATH, REF_PATH)

