from rldmuu.src.RL import MDP
import numpy as np


class ChainMDP(MDP.DiscreteMDP):
    """
    Problem where
    """

    def __init__(self, n_states=20):
        assert  n_states > 1

        n_actions = 2
        super().__init__(n_states=n_states, n_actions=n_actions)

        self.R[:] = 0.
        self.P[:] = 0.

        self.R[:, 1] = -1 / (n_states-1)
        self.R[n_states-1, 1] = 1.
        self.R[:, 0] = 1/n_states

        for i in range(self.n_states-1):
            if i > 0:
                self.P[i, 0, i-1] = 1.
            else:
                self.P[i, 0, i] = 1.

            self.P[i, 1, i+1] = 1.

        self.P[self.n_states-1, :, self.n_states-1] = 1.

if __name__ == '__main__':
    # Unit test

    p = ChainMDP()
    print(p.P, p.R)