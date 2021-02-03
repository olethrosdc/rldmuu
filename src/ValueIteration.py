import MDP
import numpy as np

## Define algorithm
def value_iteration(mdp, n_iterations, gamma):
    policy = np.zeros([mdp.n_states])
    V = np.zeros([mdp.n_states])
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    for t in range(n_iterations-1, -1, -1):
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                if (t==n_iterations-1):
                    Q[s,a,t] = mdp.get_reward(s, a)
                else:
                    P_sa = mdp.get_transition_probabilities(s, a)
                    U_next = sum(P_sa * V[:,t+1])
                    Q[s,a,t] = mdp.get_reward(s, a) + U_next
            V[s,t] = max(Q[s,:,t])
            policy[s, t] = np.argmax(Q[s,:,t])
    return policy, V, Q

n_actions = 2
n_states = 2
n_iterations = 1000
mdp = MDP.DiscreteMDP(n_states, n_actions)
policy, V, Q = backwards_induction(mdp, n_iterations)

for s in range(mdp.n_states):
    for a in range(mdp.n_actions):
        print("S:", s, "A:", a, mdp.get_transition_probabilities(s,a))


for t in range(n_iterations):
    print(policy[:,t])
        
for t in range(n_iterations):
    print(V[:,t])

    



    
