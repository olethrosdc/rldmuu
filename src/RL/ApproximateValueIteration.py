import MDP
import numpy as np
import ValueIteration

## Define algorithm
def approximate_value_iteration(mdp, n_iterations, gamma, representation_size = 4, n_samples = 4):
    policy = np.zeros([mdp.n_states])
    V = np.zeros([mdp.n_states])
    Q = np.zeros([mdp.n_actions])
    u = np.zeros(representation_size);
    ## Select a random set of sampled states.
    ## This can be improved
    SampledStates = np.random.choice(mdp.n_states,100)
    for t in range(n_iterations):
        for s in SampledStates:
            for a in range(mdp.n_actions):
                ## MC approximation of Q value
        cnt = np.zeros(representation_size);
        u = np.zeros(representation_size);
        # Minimise \sum_s |V(s) - u(s)|^2 with SGD
        for k in range(n_iterations):
            for s in SampledStates:
                s = np.random.choice(mdp.n_states)
                s_hat = s % representation_size
                u[s_hat] += 0.1 * (V[s] - u[s_hat])

    return policy, V

n_actions = 4
n_states = 64
n_iterations = 1000
gamma = 0.9
mdp = MDP.DiscreteMDP(n_states, n_actions)
a_policy, a_V= approximate_value_iteration(mdp, n_iterations, gamma, 32)
policy, V, Q= ValueIteration.value_iteration(mdp, n_iterations, gamma)
print(V)
print(a_V)
