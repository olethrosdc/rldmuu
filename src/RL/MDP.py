import numpy as np

## This a discrete MDP with a finite number of states and actions
class DiscreteMDP:
    ## initalise a random MDP with
    ## n_states: the number of states
    ## n_actions: the number of acitons
    def __init__(self, n_states, n_actions):
        self.n_states = n_states # the number of states of the MDP
        self.n_actions = n_actions # the number of actions of the MDP
        self.P = np.zeros([n_states, n_actions, n_states]) # the transition probability matrix of the MDP so that P[s,a,s'] is the probabiltiy of going to s' from (s,a)
        self.R = np.zeros([n_states, n_actions]) # the expected reward for each action and state
        # generate uniformly random transitions and 0.1 bernoulli rewards
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.P[s,a] = np.random.dirichlet(np.ones(n_states)) # generalisation of Beta to multiple outcome
                self.R[s,a] = np.round(np.random.uniform(), decimals=1)
                    
    # get the probability of next state j given current state s, action a, i.e. P(j|s,a)
    def get_transition_probability(self, state, action, next_state):
        return self.P[state,action,next_state]
    # get the vector of probabilities over next states P( . | s,a)
    def get_transition_probabilities(self, state, action):
        return self.P[state,action]
    # Get the reward for the current state action.
    # It can also be interpreted as the expected reward for the state and action.
    def get_reward(self, state, action):
        return self.R[state, action]

    

        
        
