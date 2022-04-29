import MDP
import numpy as np

## Define algorithm
def policy_iteration_iteration(mdp, n_iterations, gamma, policy = None):
    if (policy is None):
        policy = np.zeros([mdp.n_states])

    V = np.zeros([mdp.n_states])
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    for t in range(n_iterations):
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
    return policy, V, Q

n_actions = 2
n_states = 2
n_iterations = 1000
gamma = 0.9
mdp = MDP.DiscreteMDP(n_states, n_actions)
policy, V, Q = value_iteration(mdp, n_iterations, gamma)




    
