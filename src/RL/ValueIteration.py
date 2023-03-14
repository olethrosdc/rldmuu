import MDP
import numpy as np

## Define algorithm
def value_iteration(mdp, n_iterations, gamma, V = None):
    policy = np.zeros([mdp.n_states])
    assert(gamma > 0)
    assert(gamma < 1)
    if (V is None):
        V = np.zeros([mdp.n_states])
        
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    ## to fill in
    return policy, V, Q

n_actions = 2
n_states = 2
n_iterations = 1000
gamma = 0.9
mdp = MDP.DiscreteMDP(n_states, n_actions)
policy, V, Q = value_iteration(mdp, n_iterations, gamma)


print (policy)
print (V)
print (Q)


    
