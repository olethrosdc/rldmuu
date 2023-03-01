import numpy as np

## A simple utility function with the shape
## 
## -1  10
##  1   0
##
## The utility function can also be initialised otherwise 
class Utility():
    def __init__(self):
        self.U = np.zeros([2,2])
        self.U[0,0]=-1
        self.U[1,0]=1
        self.U[0,1]=10
        self.U[1,1]=0
        self.n_actions = 2
        self.n_outcomes = 2
    ## get the utility for action (a) and outcome (x)
    def get_U_ax(self, action, outcome):
        assert(action < self.n_actions)
        assert(outcome < self.n_outcomes)
        return self.U[action, outcome]
    ## set a new utility function
    def set_U(self, U):
        self.U = U
        self.n_actions = self.U.shape[0]
        self.n_outcomes = self.U.shape[1]
    ## returns the number of actions
    def get_n_actions(self):
        return self.n_actions;
    ## returns the number of outcomes
    def get_n_outcomes(self):
        return self.n_outcomes;
        
## ---------------------------- to fill in ------------------------------------
## Inputs
## utility: A Utility class instance
## action: the action
## P: an array so that P[outcome] is the probability of each outcome
## (note that this does not depend on the action -- otherwise we'd have P[action,outcome])
def expected_utility(utility, action, P):
    n_w = utility.get_n_outcomes()
    EU = 0
    for omega in range(n_w):
        EU += P[omega] * utility.get_U_ax(action, omega)
    return EU

## For a given P, get the optimal action, ie the one maximising expected utility
## U: utility class object
## P: probability of outcomes
def get_max_action(U, P):
    n_a = utility.get_n_actions()
    EU = [expected_utility(U, action, P) for action in range(n_a)]
    return np.argmax(EU)

## For a given policy assigning probability policy[action] to each action, get the minimising outcome
def get_min_outcome(U, policy):
    return None

## Find the worst-case distribution P, i.e. the one minimising
## max_\pi \sum_{\omega, a} P(\omega) U(\omega, a) \pi(a)
def get_worst_case_P(U):
    return None
## ----------------------------------------------------------------------------


## ----------------------------- unit test ------------------------------------
# initialise the variables
utility=Utility()
n_actions = 2
n_outcomes = 2
P = np.random.uniform(size=2)
P/= P.sum()

# test
a_star = get_max_action(utility, P)
print(a_star)
for a in range(n_actions):
    print(a, expected_utility(utility, a, P))
    assert(expected_utility(utility, a, P) <= expected_utility(utility, a_star, P))

