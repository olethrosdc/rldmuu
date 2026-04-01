import sys
sys.path.append('../MDP')
import ContinuousMDP
import numpy as np
import copy

## A linear value function
class LinearValueFunction:
    def __init__(self, n_dim, rate):
        self.n_dim = n_dim # n state dimension
        self.params = np.random.normal(size=n_dim) # random init
        self.rate = 1e-3 # learning rate stuff
        
    def get_value(self, state):
        return np.dot(self.params, state)
    
    ## let's say we want to take one step towards
    ## a least-squares error solution
    ## d/dw (y - f(s))^2 = 2(f(s)-y) df/dw
    ## We also know that d/dw_i \sum_j w_j s_j = s_i
    ## So d/dw (y - f(s))^2 = s (f(s) - y)
    ## Crucially, the target includes no parameters
    def update(self, state, target):
        cost_gradient = [self.get_value(state) - target]
        gradient = state * cost_gradient
        self.params -= self.rate * gradient

## Define algorithm
def approximate_value_iteration(mdp, gamma, v_hat, n_samples = 1000, n_iterations = 100, n_next_samples=10):
    ## Select a random set of sampled states.
    ## n_samples : number of state seamples
    ## n_dim: state dimensionality
    n_dim = mdp.n_states
    SampledStates = np.random.normal(size = [n_samples, n_dim])
    
    for t in range(n_iterations):
        v_old = copy.copy(v_hat) # copy parameters so that we avoid instability
        for s in SampledStates:
            v_s = v_hat.get_value(s)
            q = np.zeros(mdp.n_actions)
            for a in range(mdp.n_actions):
                next_util = 0
                for k in range(n_next_samples):
                    next_util += v_old.get_value(mdp.generate_state(s,a))
                q[a] = mdp.get_reward(s, a) + gamma * next_util / n_next_samples
            v_target = max(q)
            v_hat.update(s, v_target)
    return v_hat


## Define algorithm
def fitted_q_iteration(mdp, gamma, q_hat, n_dim, n_samples, n_iterations, n_next_samples):
    ## Select a random set of sampled states.
    ## n_samples : number of state seamples
    ## n_dim: state dimensionality
    SampledStates = np.random.normal(size = [n_samples, n_dim])
    
    for t in range(n_iterations):
        q_old = q_hat.deepcopy() # copy parameters so that we avoid instability
        for s in SampledStates:
            v_s = v_hat.get_value(s)
            q_target = np.zeros(mdp.n_actions)
            for a in range(mdp.n_actions):
                next_util = 0
                for k in range(n_next_samples):
                    next_s = mdp.generate_state(s,a)
                    #next_util += max(q_old.get_values(next_s)) # max_a Q(s', a)
                    next_util += max(monte_carlo_sampler(mdp, next_s, q_old)) # max_a Q(s', a)
                q_target[a] = mdp.get_reward(s, a) + gamma * next_util / n_next_samples
            q_hat.update(s, q_target)
    return q_hat

def monte_carlo_sampler(mdp, gamma,  starting_state, q_hat, n_samples, T):
    for a in range(mdp.n_actions):
        for k in range(n_samples):
            state = starting_state
            next_action = a
            discount = 1
            for t in range(T):
                next_state = mdp.generate_state(state, next_action)
                q_sample[a] += discount * mdp.get_reward(state, next_action)
                next_action = argmax(q_hat.get_values(next_state))
                discount *= gamma
    return q_sample/n_samples

n_actions = 4
n_states = 64
n_iterations = 1000
gamma = 0.9
mdp = ContinuousMDP.LinearMDP(n_states, n_actions)
approximator = LinearValueFunction (n_states, 0.001)
a_policy, a_V= approximate_value_iteration(mdp, 0.9, approximator)
policy, V, Q= ValueIteration.value_iteration(mdp, n_iterations, gamma)
print(V)
print(a_V)
