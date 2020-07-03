import numpy as np
import math

'''
lidar data test
single destination
'''


def value_iteration(shape, rewards, discount, terminal):

    [H, W, N_STATES, N_ACTIONS] = shape

    dx = [0, 0, -1, 1, -1, 1, -1, 1]
    dy = [-1, 1, 0, 0, -1, -1, 1, 1]
    # ------------------------------->x
    # |           dx    dy     action
    # | up        0     -1        0
    # | down      0     1         1
    # | left      -1    0         2
    # | right     1     0         3
    # | up-left   -1    -1        4
    # | up-right  1     -1        5
    # | down-left -1    1         6
    # | down-right 1    1         7
    # y

    v = np.nan_to_num(np.ones(N_STATES) * float("-inf"))
    Q = np.zeros((N_STATES, N_ACTIONS))
    policy = np.zeros((N_STATES, N_ACTIONS))

    len_terminal = len(terminal)
    for i in range(105):
        vt = np.nan_to_num(np.ones(N_STATES) * float("-inf"))
        '''multiple terminals'''
        for j in range(len_terminal):
            vt[terminal[j]] = np.log(1.0/len_terminal)
        # vt[terminal] = 0
        for s in range(N_STATES):
            x, y = int_to_point(W, s)
            for a in range(N_ACTIONS):
                x1 = x + dx[a]
                y1 = y + dy[a]
                if (x1 not in range(W)) or (y1 not in range(H)):
                    # Q[s, a] = np.nan_to_num(float("-inf"))
                    continue
                else:
                    s1 = point_to_int(W, x1, y1)
                    # Q[s, a] = rewards[s] + discount * v[s1]
                    vt[s] = softmax(vt[s], rewards[s] + discount * v[s1])
        vt = vt - np.log(8.0)
        # print(np.max(np.abs(vt-v)))
        v = vt

    # v[terminal] = 0
    for j in range(len_terminal):
        v[terminal[j]] = np.log(1.0 / len_terminal)

    for s in range(N_STATES):
        x, y = int_to_point(W, s)
        for a in range(N_ACTIONS):
            x1 = x + dx[a]
            y1 = y + dy[a]
            if (x1 not in range(W)) or (y1 not in range(H)):
                Q[s, a] = np.nan_to_num(float("-inf"))
                continue
            else:
                s1 = point_to_int(W, x1, y1)
                Q[s, a] = rewards[s] + discount * v[s1]
        Q_max = Q[s, :].max()
        Q[s, :] -= Q_max
        policy[s, :] = np.exp(Q[s, :]) / np.exp(Q[s, :]).sum()

    return v, policy


def normalize(vals):
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val) / (max_val - min_val)


def softmax(x1, x2):
    maxx = max(x1, x2)
    minn = min(x1, x2)
    return maxx + np.log(1 + np.exp(minn - maxx))


def int_to_point(W, i):
    return i % W, i // W


def point_to_int(W, x, y):
    return y*W + x
