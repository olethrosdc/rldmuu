import numpy as np
import MDP
import ValueIteration

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

    # get marginal transition probability
    def get_marginal_transition_probability(self, state, action, next_state):

    # get the vector P( . | s,a)
    def get_marginal_transition_probabilities(self, state, action):
        
    # get the reward for the current state action
    def get_expected_reward(self, state, action):

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
                mdp.P[s,a] = np.random.dirichlet(self.alpha[s, a])
                mdp.R[s,a] = np.random.beta(self.reward_alpha[s,a], self.reward_beta[s,a])
        return mdp


## A simple model-base reinforcement learning algorithm
class ExpectedMDPHeuristic:
    def __init__(self, n_states, n_actions, discount=0.9, alpha = 0.01, epsilon=0.1):
        self.n_iterations = 10
        self.n_actions = n_actions
        self.n_states = n_states
        self.belief = DiscreteMDPBelief(n_states = n_states, n_actions = n_actions)
        self.discount = discount
        self.alpha = 0.01
        self.epsilon = 0.1
        self.V = None
        self.calculate_policy()
    def calculate_policy(self):
        mdp = self.belief.get_mean_MDP()
        self.policy, self.V, _ = ValueIteration.value_iteration(mdp, self.n_iterations, self.discount, self.V)
    def act(self):
        self.calculate_policy()
        self.epsilon = 1 / (1 / self.epsilon + 0.5)
        if (np.random.uniform() < self.epsilon):
            return np.random.choice(self.n_actions)
        return int(self.policy[self.state])
    
    def update(self, action, reward, state):
        self.belief.update(state=self.state, action=action, next_state=state, reward=reward)
        self.state = state

    def reset(self, state):
        self.calculate_policy()
        self.state = state
        #print(self.policy)


    
class SampleBasedRL:
    def __init__(self, n_states, n_actions, discount=0.9, alpha = 0.01, epsilon=0.1):
        self.n_iterations = 10
        self.n_actions = n_actions
        self.n_states = n_states
        self.belief = DiscreteMDPBelief(n_states = n_states, n_actions = n_actions)
        self.discount = discount
        self.alpha = 0.01
        self.epsilon = 0.1
        self.V = None
        self.calculate_policy()
    def calculate_policy(self):
        mdp = self.belief.get_MDP_sample()
        self.policy, self.V, _ = ValueIteration.value_iteration(mdp, self.n_iterations, self.discount, self.V)
    def act(self):
        self.calculate_policy()
        return int(self.policy[self.state])
    
    def update(self, action, reward, state):
        self.belief.update(state=self.state, action=action, next_state=state, reward=reward)
        self.state = state

    def reset(self, state):
        self.calculate_policy()
        self.state = state

        #print(self.policy)

        
