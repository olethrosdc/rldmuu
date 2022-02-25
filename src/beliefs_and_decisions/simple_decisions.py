import numpy as np

## A simple utility function with the shape
## 
## -1  10
##  1   0
##
## In this utility function, 
class Utility():
    def __init__(self):
        self.U = np.zeros([2,2])
        self.U[0,0]=-1
        self.U[1,0]=1
        self.U[0,1]=10
        self.U[1,1]=0
    def U(self, action, outcome):
        return self.U[action, outcome]

## to fill in

## Inputs
## utility: A Utility class instance
## action: the action
## P: an array so that P[outcome] is the probability of each outcome
def expected_utility(utility, action, P):
    return 0

## For a given P, get the optimal action, ie the one maximising expected utility
def get_max_action(U, P):
    return 0

## For a given policy assigning probability policy[action] to each action, get the minimising outcome
def get_min_outcome(U, policy):
    return None

## Find the worst-case distribution P, i.e. the one minimising
## max_\pi \sum_{\omega, a} P(\omega) U(\omega, a) \pi(a)
def get_worst_case_P(U):
    return None

# initialise the variables
utility=Utility()
n_actions = 2
n_outcomes = 2
P = np.random.uniform(size=2)
P/= P.sum()

# test
a_star = get_max_action(utility, P)
for a in range(n_actions):
    assert(expected_utility(utility, a, P) <= expected_utility(utility, a_star, P))

