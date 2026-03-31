import MDP
import mdp_examples
import numpy as np

## Define algorithm
def value_iteration(mdp, n_iterations, gamma, V = None):
    policy = np.zeros([mdp.n_states])
    assert(gamma > 0)
    assert(gamma < 1)
    # expected utility of states
    if (V is None):
        V = np.zeros([mdp.n_states])

    # expected utility of state-action pairs
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    ## to fill in
    for n in range(n_iterations):
        a = policy(s)
        # rewards
        s_next = mdp.generate_next_state(s, a)
        Q[s,a] += 0.1 * (mdp.get_reward(s, a) + gamma * max(Q[s_next,:] - Q[s,a]))
        s = s_next
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


    
