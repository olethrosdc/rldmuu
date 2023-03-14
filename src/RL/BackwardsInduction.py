import MDP
import numpy as np

## Define algorithm
## policy[state, time] gives you index of the action played at any state for any time.
def backwards_induction(mdp, policy, T):
    V = np.zeros([mdp.n_states, T])
    Q = np.zeros([mdp.n_states, mdp.n_actions, T])
    ## to fill in
    # First fill in the value at time T:
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions): 
            Q[state, action, T - 1] = mdp.get_reward(state, action)
            V[state, T-1] = max(Q[state, :, T-1])
            policy[state, T-1] = np.argmax(Q[state, :, T-1])

    for k in range(T - 1):
        t = T - 2 - k
        for state in range(mdp.n_states):
            for action in range(mdp.n_actions):
                action = policy[state, t]
                reward = mdp.get_reward(state, action)
                Q[state, action, t] = reward
                for next_state in range(mdp.n_states):
                    Q[state, action, t] += mdp.get_transition_probability(state, action, next_state) * V[next_state, t+1]
            V[state, t] = max(Q[state, :, t])
            policy[state, t] = np.argmax(Q[state, :, t])
    return policy, V, Q

n_actions = 2
n_states = 2
T = 2
mdp = MDP.DiscreteMDP(n_states, n_actions)

policy = np.zeros([n_states, T], int) #this plays action 0 all the time
policy, V, Q = backwards_induction(mdp,  policy, T)


for s in range(mdp.n_states):
    for a in range(mdp.n_actions):
        print("S:", s, "A:", a, mdp.get_transition_probabilities(s,a))


for t in range(T):
    print(policy[:,t])
        
for t in range(T):
    print(V[:,t])

    



    
