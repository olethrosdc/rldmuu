## A simple probability class

import numpy as np

class Probability:
    def __init__(self, n_outcomes=2, P=None):
        if (P):
            self.P = P
            self.n_outcomes = P.shape[0]
        else:
            self.P = np.ones(n_outcomes) / n_outcomes
            self.n_outcomes = n_outcomes
            
    def get_prob(self, x):
        return self.P[x]

    def get_sample_space(self):
        return range(self.n_outcomes)
    
    
## Inputs
## - a finite probability measure P implementing get_prob
## - a random variable f on P
def calculate_expectation(P, f):
    ## {f(x) | x \in S\}
    return sum([P.get_prob(x) * f(x) for x in P.get_sample_space()])
    ## FILL IN
    
## A real-valued random variable mapping from P's sample space to a real number
def function(x):
    return x 
    ## FILL IN

P = Probability(n_outcomes = 5)
Ef = calculate_expectation(P, function)
print(Ef) # print the expected value of the functio under P





