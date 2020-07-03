from deep_maxent_irl import *
import input_data as input_data


def main():
    # hyper parameters
    H = 48
    W = 128
    N_STATES = H*W
    N_ACTIONS = 8
    SHAPE = [H, W, N_STATES, N_ACTIONS]
    DISCOUNT = 1
    LEARNING_RATE_BASE = 0.001

    DECAY_STEPS = 500
    DECAY_RATE = 0.9
    LEARNING_RATE = [LEARNING_RATE_BASE, DECAY_STEPS, DECAY_RATE]

    N_ITERATION = 1500
    GRAPH_SAVE_INTERVAL = 5

    terminal, img, traj, ref = input_data.read_img(SHAPE)

    rewards = deep_maxent_irl(SHAPE, img, DISCOUNT, terminal, traj, ref, LEARNING_RATE,
                              N_ITERATION, GRAPH_SAVE_INTERVAL)

if __name__ == "__main__":
    main()
