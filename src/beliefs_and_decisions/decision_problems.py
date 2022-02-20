## A simple utility function with the shape
## 
## -1  10
##  1   0
class Utility():
    def __init__(self):
        self.U = np.zeros(2,2)
        self.U[0,0]=-1
        self.U[1,0]=1
        self.U[0,1]=10
        self.U[1,1]=0
    def U(self, action, outcome):
        return self.U[action, outcome]

## Inputs
## utility: A Utility class instance
## action: the action
## P: an array so that P[outcome] is the probability of each outcome
def expected_utilility(utility, action, P):


## For a given P, get the optimal action, ie the one maximising expected utility
def get_max_action(U, P):

## For a given policy assigning probability policy[action] to each action, get the minimising outcome
def get_min_outcome(U, policy):


## Find the worst-case distribution P, i.e. the one minimising
## max_\pi \sum_{\omega, a} P(\omega) U(\omega, a) \pi(a)
def get_worst_case_P(U):
    
