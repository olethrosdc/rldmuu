import numpy as np

## This a discrete MDP with a finite number of states and actions
class DiscreteMDP:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        P = np.zeros(n_states, n_actions, n_states)
        R = np.zeros(n_states, n_actions)
        # generate uniformly random transitions and 0.1 bernoulli rewards
        for s in range(self.n_states):
            for a in range(self.n_actions):
                P[s,a] = np.random.dirichlet(np.ones(n_states)) # generalisation of Beta to multiple outcome
                R[s,a] = np.random.uniform()
                if (R[s,a] > 0.9):
                    R[s,a] = 1
                else:
                    R[s,a] = 0
                    
    # get a single P(j|s,a)
    def get_transition_probability(self, state, action, next_state):
        return P[state,action,next_state]
    # get the vector P( . | s,a)
    def get_transition_probabilities(self, state, action):
        return P[state,action]
    def get_reward(self, state, action):
        return R[state, action]
    
    

        
        
