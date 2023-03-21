import MDP
import mdp_examples
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
    for n in range(n_iterations):
        V_old = V.copy()
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                Q[s, a] = mdp.get_reward(s, a)
                for j in range(mdp.n_states):
                    Q[s,a] += gamma * mdp.get_transition_probability(s,a,j) * V_old[j]
            V[s] = max(Q[s,:])
            policy[s] = np.argmax(Q[s,:])
    return policy, V, Q

n_actions = 2
n_states = 2
n_iterations = 1000
gamma = 0.9
mdp = mdp_examples.ChainMDP()
policy, V, Q = value_iteration(mdp, n_iterations, gamma)


print (policy)
print (V)
print (Q)


    
