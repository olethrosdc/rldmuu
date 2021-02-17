import MDP
import numpy as np

## Define algorithm
def value_iteration(mdp, n_iterations, gamma):
    policy = np.zeros([mdp.n_states])
    V = np.zeros([mdp.n_states])
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    for t in range(n_iterations):
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                P_sa = mdp.get_transition_probabilities(s, a)
                U_next = sum(P_sa * V[:])
                Q[s,a] = mdp.get_reward(s, a) + gamma * U_next
            V[s] = max(Q[s,:])
            policy[s] = np.argmax(Q[s,:])
            #print(V)
    return policy, V, Q

n_actions = 2
n_states = 2
n_iterations = 1000
gamma = 0.9
mdp = MDP.DiscreteMDP(n_states, n_actions)
policy, V, Q = value_iteration(mdp, n_iterations, gamma)




    
