import MDP
import numpy as np


class ChainMDP(MDP.DiscreteMDP):
    """
    Problem where we need to take the same action n_states-1 time in a row to get a highly rewarding state
    The optimal policy greatly depends on the discount factor we choose.
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


class SecretaryMDP(MDP.DiscreteMDP):
    """
    Implementation of the basic secretary problem.
    """

    def __init__(self, n_candidates=10):

        super().__init__(n_states=2*n_candidates+1, n_actions=2)

        self.P[:] = 0.
        self.R[:] = 0.

        self.QUIT = 1
        self.CONTINUE = 0
        self.end_state = self.n_states-1

        # If we quit, we go to the end state
        self.P[:, self.QUIT, self.end_state] = 1.

        # If we don't quit, move to the next candidate
        best_sofar_idx = np.arange(0, self.n_states-1, 2) # pair idxs are states where we have the best candidate seen so far.
        regular_candidate_idx = np.arange(1, self.n_states-1, 2) # odd idxs are for states where we have no the best candidate

        for i, idx in enumerate(best_sofar_idx):
            if i == n_candidates-1:
                # Last candidate
                self.P[idx, self.CONTINUE, self.end_state] = 1
            else:
                # Go from a best so far individual to an even better individual
                self.P[idx, self.CONTINUE, idx+2] = 1 / (2 + i)
                # Go from a best so far individual to a regular individual
                self.P[idx, self.CONTINUE, idx+3] = (1 + i) / (2 + i)

        for i, idx in enumerate(regular_candidate_idx):
            if i == n_candidates-1:
                # Last candidate
                self.P[idx, self.CONTINUE, self.end_state] = 1
            else:
                # Go from a regular individual to best so far individual
                self.P[idx, self.CONTINUE, idx + 3] = 1 / (2 + i)
                # Go from a regular to a regular
                self.P[idx, self.CONTINUE, idx + 2] = (1 + i) / (2 + i)

        # We are only rewarded for getting a best so far individual
        self.R[best_sofar_idx, self.QUIT] = [n_seen / n_candidates for n_seen in range(n_candidates)]




if __name__ == '__main__':
    # Unit test

    p = ChainMDP()
    #print(p.P, p.R)

    p = SecretaryMDP()
    print(p.P, p.R)

