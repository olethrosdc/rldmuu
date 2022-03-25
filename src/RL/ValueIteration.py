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
    for t in range(n_iterations):
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                Q[s,a] = mdp.get_reward(s,a) + gamma * sum([V[s2] * mdp.get_transition_probability(s,a,s2) for s2 in range(mdp.n_states)])
            V[s] = max(Q[s,:])
            print(V)
    for s in range(mdp.n_states):
        policy[s] = np.argmax(Q[s,:])
        
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


    
