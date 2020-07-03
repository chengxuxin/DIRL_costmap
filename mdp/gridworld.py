'''
相比gridworld 转移矩阵的设置为 3 actions
gridworld2 转移矩阵的设置为 8 actions
'''

import numpy as np


class Gridworld(object):

    def __init__(self, H, W, wind):
        self.H = H
        self.W = W
        self.actions = ((0, -1), (0, 1), (-1, 0), (1, 0), (1, -1), (-1, -1), (-1, 1), (1, 1))
        # 0上 1下 2左 3右 4上右 5上左 6下左 7下右
        self.n_actions = len(self.actions)
        print(self.n_actions)
        self.grid_size = self.H * self.W
        self.n_states = H * W
        self.wind = wind

        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])

    def int_to_point(self, i):
        return i % self.W, i // self.W

    def point_to_int(self, p):
        return p[1]*self.W + p[0]

    def neighbouring(self, i, k):
        if (abs(i[0] - k[0]) <= 1) and (abs(i[1] - k[1]) <= 1):
            return 1.0
        else:
            return 0.0

    def _transition_probability(self, i, j, k):
        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind/self.n_actions

        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        if (xi, yi) in {(0, 0), (self.W-1, self.H-1),
                        (0, self.H-1), (self.W-1, 0)}:
            if not (0 <= xi + xj < self.W and
                    0 <= yi + yj < self.H):
                return 1 - self.wind + 5*self.wind/self.n_actions
            else:
                return 5*self.wind/self.n_actions
        else:
            if xi not in {0, self.W-1} and yi not in {0, self.H-1}:
                return 0.0

            if not (0 <= xi + xj < self.W and
                    0 <= yi + yj < self.H):
                return 1 - self.wind + 3*self.wind/self.n_actions
            else:
                return 3*self.wind/self.n_actions
