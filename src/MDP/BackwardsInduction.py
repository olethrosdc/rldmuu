import MDP
import numpy as np

## Define algorithm
## policy[state, time] gives you index of the action played at any state for any time.
def backwards_induction(mdp, policy, T):
    V = np.zeros([mdp.n_states, T])
    Q = np.zeros([mdp.n_states, mdp.n_actions, T])
    ## to fill in
    # First fill in the value at time T:

    # Then go back and fill all other state values

    #
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

    



    
