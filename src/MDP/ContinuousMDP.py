import numpy as np

## This a continuous MDP with a finite number of states and actions
## This is just the API
class ContinuousMDP:
    ## initalise a random MDP with
    ## n_states: the number of states
    ## n_actions: the number of actions
    ## Optional arguments:
    ## P: the state-action-state transition matrix so that P[s,a,s_next] is the probability of s_next given the current state-action pair (s,a)
    ## R: The state-action reward matrix so that R[s,a] is the reward for taking action a in state s.
    def __init__(self, n_states, n_actions, P = None, R = None):
        self.n_states = n_states # the dimension of states of the MDP
        self.n_actions = n_actions # the number of actions of the MDP
        self.state = np.random.normal(size=n_states)
        self.A = np.random.uniform(size =[n_states, n_actions, n_states])
    # generate a new state
    def generate_state(self, state, action):
        pass
    
    # Get the reward for the current state action.
    # It can also be interpreted as the expected reward for the state and action.
    def get_reward(self, state, action):
        pass


class LinearMDP:
    ## initalise a random MDP with
    ## n_states: the number of states
    ## n_actions: the number of actions
    def __init__(self, n_states, n_actions, P = None, R = None):
        self.n_states = n_states # the dimension of states of the MDP
        self.n_actions = n_actions # the number of actions of the MDP
        self.P = np.random.uniform(size=[n_actions, n_states, n_states])
        self.R = np.random.uniform(size=[n_actions, n_states])
        self.state = np.random.uniform(size=n_states)
        self.sigma = 0.1
    # generate a new state
    def generate_state(self, state, action):
        return self.P[action] @ self.state + self.sigma * np.random.normal(size=self.n_states)
    
    # Get the reward for the current state action. Here it is just a linear function
    def get_reward(self, state, action):
        return self.R[action] @ self.state


    
        
        
