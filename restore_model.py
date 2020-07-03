import tensorflow as tf
import input_data
import matplotlib.pyplot as plt
import img_utils
import numpy as np
import mdp.value_iteration4 as vi
import mdp.gridworld2 as gridworld


H = 24
W = 64
N_STATES = H*W
TRAJ_LEN = 4

def int_to_point(i):
    return i % W, i // W

__, img, _ = input_data.read_img(H, W)
input_img = tf.placeholder(tf.float32, [None, H, W, 3])

sess = tf.Session()

# load meta graph and restore weights
saver = tf.train.import_meta_graph('/Users/David/Desktop/model/model24*64/ckpt/model119.ckpt.meta')
saver.restore(sess, '/Users/David/Desktop/model/model24*64/ckpt/model119.ckpt')

# get all tensors. not necessary
graph = tf.get_default_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
print(tensor_name_list)

inimg = graph.get_tensor_by_name("input_img:0")
feed_dict = {inimg: img}

op_to_restore = graph.get_tensor_by_name("rewards:0")

r = sess.run(op_to_restore, feed_dict)
r = -r

# get trajectory from trained reward map
r = np.reshape(r, [N_STATES, ])
r = np.exp(r)
r = r / np.max(r)
for i in range(N_STATES):
    if r[i] > 0.9:
        r[i] = 1
    if r[i] < 0.8:
        r[i] = 0.0
'''
DISCOUNT = 0.9
N_ITERATION = 1500
WIND = 0.0
VALUE_ITERATION_ERROR = 0.001
DETERMINISTIC = False

# gridworld
gw = gridworld.Gridworld(H, W, WIND)
P_a = gw.transition_probability
N_STATES, N_ACTIONS, _ = np.shape(P_a)
value, policy = vi.value_iteration(P_a, r, DISCOUNT, VALUE_ITERATION_ERROR, DETERMINISTIC)
img, trajs = input_data.read_img(H, W)

traj_demo = np.zeros([H, W])
traj = np.zeros(TRAJ_LEN)
traj[0] = trajs[0]

a_flg = 0
act = np.zeros(N_STATES)

for s in range(H * W):
    # get optimal action
    m = -999999
    for a in range(N_ACTIONS):
        if policy[s][a] < m:
            a_flg = a
            m = policy[s][a]
    act[s] = a_flg


for i in range(1, TRAJ_LEN, 1):
    for s1 in range(N_STATES):
        s = int(traj[i - 1])
        action = int(act[s])
        if P_a[s][action][s1] == 1:
            traj[i] = s1

for j in range(0, TRAJ_LEN, 1):
    x, y = int_to_point(traj[j])
    x = int(x)
    y = int(y)
    traj_demo[y][x] = 1

for i in range(H):
    for j in range(W):
        if r[i+j] > 20:
            r[i+j] = 20
        r[i+j] = 0
'''
plt.figure(figsize=(H, W))
img_utils.heatmap2d(np.reshape(r, (H, W)), 'Reward Map', block=False, text=False)
plt.show()
'''
plt.imshow(traj_demo)
plt.show()
plt.figure(figsize=(H, W))
img_utils.heatmap2d(np.reshape(value, (H, W)), 'Value', block=False, text=False)
plt.show()'''


