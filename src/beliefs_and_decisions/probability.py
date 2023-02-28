## A simple probability class

import numpy as np

class ProbabilitySpace:
    def __init__(self, n_outcomes=2, P=None):
        if P is not None:
            self.P = np.array(P)
            self.n_outcomes = self.P.shape[0]
        else:
            self.P = np.ones(n_outcomes) / n_outcomes
            self.n_outcomes = n_outcomes
            
    def get_prob(self, x):
        return self.P[x]

    def get_sample_space(self):
        return np.arange(self.n_outcomes)
    
    
## Inputs
## - a finite probability measure P implementing get_prob
## - a random variable f on P
## Outputs
## - E_P[f] = \sum_{x \in S} f(x) P(x)
def calculate_expectation(P, f):
    ## FILL IN
    
## A real-valued random variable mapping from P's sample space to a real number
def function(x):
    ## FILL IN

P = ProbabilitySpace(n_outcomes = 5)
Ef = calculate_expectation(P, function)
print(Ef) # print the expected value of the function under P






