import numpy as np
import MDP
## This a discrete MDP with a finite number of states and actions
class DiscreteMDPBelief:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        ## Dirichlet parameters
        self.alpha = np.ones([n_states, n_actions, n_states])
        ## Reward parameters
        self.reward_alpha = np.ones([n_states, n_actions])
        self.reward_beta = np.ones([n_states, n_actions])

    # Calculate the dirichlet and beta posteriors from the data point
    def update(self, state, action, next_state, reward):
        self.alpha[state, action, next_state] += 1
        self.reward_alpha[state, action] += reward
        self.reward_beta[state, action] += (1 - reward)
    # get marginal transition probability
    def get_marginal_transition_probability(self, state, action, next_state):
        return self.alpha[state, action, next_state] / self.alpha[state, action].sum()
    # get the vector P( . | s,a)
    def get_marginal_transition_probabilities(self, state, action):
        return self.alpha[state, action] / self.alpha[state, action].sum()
    # get the reward for the current state action
    def get_expected_reward(self, state, action):
        return self.reward_alpha[state, action] / (self.reward_alpha[state, action] + self.reward_beta[state, action])

    ## Get an MDP corresponding to the marginal probabilities
    def get_mean_MDP(self):
        mdp = MDP.DiscreteMDP(self.n_states, self.n_actions)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                ## Sample transitions from the Dirichlet
                mdp.P[s,a] = self.get_marginal_transition_probabilities(s, a)
                mdp.R[s,a] = self.get_expected_reward(s,a)
        return mdp

    ## Get a Random MDP
    def get_MDP_sample(self):
        mdp = MDP.DiscreteMDP(self.n_states, self.n_actions)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                ## Sample transitions from the Dirichlet
                mdp.P[s,a] = np.random.dirichlet(self.alpha([s, a]))
                mdp.R[s,a] = np.random.beta(self.reward_alpha[s,a], self.reward_beta[s,a])
        return mdp

        

    

        
        
