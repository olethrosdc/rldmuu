import MDP
import mdp_examples
import numpy as np

def policy_evaluation(mdp, gamma, policy):
    # build the matrix of next states
    P = np.zeros([mdp.n_states, mdp.n_states])
    R = np.zeros(mdp.n_states)
    for s in range(mdp.n_states):
        P[s,:] = mdp.get_transition_probabilities(s, policy[s])
        R[s] = mdp.R[s, policy[s]]
    I = np.eye(mdp.n_states)
    ## A @ y equivalent to np.matmul(A, Y)
    ## If y is a vector in $R^n$, then it becomes a [n x 1] matrix
    ## So if A is $R^{n \times n}$ then $A r \in R^n$.
    return np.linalg.inv(I - gamma * P) @ R 

def policy_evaluation_dp(mdp, gamma, policy):
    # build the matrix of next states
    V = np.zeros([mdp.n_states])
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    ## to fill in
    for n in range(1000):
        V_old = V.copy()
        for s in range(mdp.n_states):
            V[s] = mdp.get_reward(s, policy[s])
            for j in range(mdp.n_states):
                V[s] += gamma * mdp.get_transition_probability(s, policy[s],j) * V_old[j]

    return V

## Define algorithm
def policy_iteration(mdp, n_iterations, gamma, policy = None):
    if (policy is None):
        policy = np.zeros([mdp.n_states], int)
    V = np.zeros([mdp.n_states])
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    old_policy = policy.copy()
    # Evaluation Step
    for iteration in range(n_iterations):
        # Evaluate policy
        V = policy_evaluation(mdp, gamma, policy)
        # Improve policy
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                Q[s, a] = mdp.get_reward(s, a)
                for j in range(mdp.n_states):
                    Q[s,a] += gamma * mdp.get_transition_probability(s,a,j) * V[j]
            policy[s] = np.argmax(Q[s,:])
            
        if (sum(abs(policy - old_policy))<1):
            break
        else:
            old_policy = policy.copy()
    # fill in
    return policy, V, Q

n_actions = 2
n_states = 2
n_iterations = 1000
gamma = 0.9
mdp = mdp_examples.ChainMDP(5)
policy, V, Q = policy_iteration(mdp, n_iterations, gamma)
#print(policy)
print(V)
#print(Q)






    
