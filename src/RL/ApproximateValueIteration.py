import sys
sys.path.append('../MDP')
import ContinuousMDP
import numpy as np
import copy
import matplotlib.pyplot as plt

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

## A linear Q function
class LinearQFunction:
    def __init__(self, n_dim, n_actions, rate):
        self.n_dim = n_dim # n state dimension
        self.n_actions = n_actions # n state dimension
        self.params = np.random.normal(size=[n_dim, n_actions]) # random init
        self.rate = 1e-3 # learning rate stuff
        
    def get_value(self, state, action):
        return np.dot(self.params[:, action], state)

    def get_values(self, state):
        values = [np.dot(self.params[:,action], state) for action in range(self.n_actions)]
        return values
    
    ## let's say we want to take one step towards
    ## a least-squares error solution
    ## d/dw (y - f(s))^2 = 2(f(s)-y) df/dw
    ## We also know that d/dw_i \sum_j w_j s_j = s_i
    ## So d/dw (y - f(s))^2 = s (f(s) - y)
    ## Crucially, the target includes no parameters
    def update(self, state, action, target):
        cost_gradient = [self.get_value(state, action) - target]
        gradient = state * cost_gradient
        self.params[:,action] -= self.rate * gradient

        
## Define algorithm
def approximate_value_iteration(mdp, gamma, v_hat, n_samples = 1000, n_iterations = 100, n_next_samples=100):
    ## Select a random set of sampled states.
    ## n_samples : number of state seamples
    ## n_dim: state dimensionality
    n_dim = mdp.n_states
    SampledStates = np.random.normal(size = [n_samples, n_dim])
    print("Starting AVI")
    for t in range(n_iterations):
        print("Iteration ", t)
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
    plt.plot([v_hat.get_value(s) for s in SampledStates])
    plt.show()
    return v_hat


## Define algorithm
def fitted_q_iteration(mdp, gamma, q_hat, n_samples = 1000, n_iterations = 100, n_next_samples = 10, n_mc_samples=10):
    ## Select a random set of sampled states.
    ## n_samples : number of state seamples
    ## n_dim: state dimensionality
    n_dim = mdp.n_states
    SampledStates = np.random.normal(size = [n_samples, n_dim])
    print("Starting Fitted Q Iteration")
    for t in range(n_iterations):
        print(t, "/", n_iterations)
        q_old = copy.copy(q_hat) # copy parameters so that we avoid instability
        for s in SampledStates:
            q_target = np.zeros(mdp.n_actions)
            for a in range(mdp.n_actions):
                next_util = 0
                for k in range(n_next_samples):
                    next_s = mdp.generate_state(s,a)
                    next_util += max(monte_carlo_sampler(mdp, next_s, gamma, q_old, n_mc_samples, int(np.ceil(1/(1 - gamma))))) # max_a Q(s', a)
                q_target[a] = mdp.get_reward(s, a) + gamma * next_util / n_next_samples
                q_hat.update(s, a, q_target[a])
    return q_hat

def monte_carlo_sampler(mdp, starting_state, gamma, q_hat, n_samples, T):
    q_sample = np.zeros(mdp.n_actions)
    for a in range(mdp.n_actions):
        for k in range(n_samples):
            state = starting_state
            discount = 1
            action = a
            for t in range(T):
                next_state = mdp.generate_state(state, action)
                q_sample[a] += discount * mdp.get_reward(state, action)
                state = next_state
                action = np.argmax(q_hat.get_values(state))
            discount *= gamma
        q_sample[a] /= n_samples
    return q_sample

n_actions = 4
n_states = 64
n_iterations = 1000
gamma = 0.9
mdp = ContinuousMDP.LinearMDP(n_states, n_actions)
state_approximator = LinearValueFunction (n_states, 0.001)
#V_hat= approximate_value_iteration(mdp, 0.9, state_approximator, n_samples =10 )
Q_approximator = LinearQFunction(n_states, n_actions, 0.001)
Q_hat= fitted_q_iteration(mdp, 0.9, Q_approximator, n_samples=10)
print(V)
print(a_V)
