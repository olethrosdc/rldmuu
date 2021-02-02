import MDP
import numpy as np

## Define algorithm
def backwards_induction(mdp, T):
    policy = np.zeros(T)
    V = np.zeros([mdp.n_states, T])
    Q = np.zeros([mdp.n_states, mdp.n_actions, T])
    for t in range(T-1, 0, -1):
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                if (t==T-1):
                    Q[s,a,t] = mdp.get_reward(state, action)
                else:
                    P_sa = mdp.get_transition_probabilities(state, action)
                    U_next = sum(P_sa * V[:,t+1])
                    Q[s,a,t] = mdp.get_reward(state, action) + U_next
                V[s,t] = max(Q[s,:,t])
                policy[t] = np.argmax(Q[s,:,t])
    return policy, V, Q

n_actions = 2
n_states = 5
T = 10
mdp = MDP.DiscreteMDP(n_states, n_actions)
policy, V, Q = backwards_induction(mdp, T)



    
