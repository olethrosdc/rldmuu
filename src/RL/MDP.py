import numpy as np

## This a discrete MDP with a finite number of states and actions
class DiscreteMDP:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states # the number of states of the MDP
        self.n_actions = n_actions # the number of actions of the MDP
        self.P = np.zeros([n_states, n_actions, n_states]) # the transition probability matrix of the MDP so that P[s,a,s'] is the probabiltiy of going to s' from (s,a)
        self.R = np.zeros([n_states, n_actions]) # the expected reward for each action and state
        # generate uniformly random transitions and 0.1 bernoulli rewards
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.P[s,a] = np.random.dirichlet(np.ones(n_states)) # generalisation of Beta to multiple outcome
                self.R[s,a] = np.random.uniform()
                if (self.R[s,a] > 0.9):
                    self.R[s,a] = 1
                else:
                    self.R[s,a] = 0
                    
    # get a single P(j|s,a)
    def get_transition_probability(self, state, action, next_state):
        return self.P[state,action,next_state]
    # get the vector P( . | s,a)
    def get_transition_probabilities(self, state, action):
        return self.P[state,action]
    # get the reward for the current state action
    def get_reward(self, state, action):
        return self.R[state, action]

    

        
        
