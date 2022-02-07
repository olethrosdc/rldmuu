import numpy as np

## Calculate expectation
## Inputs
## - P : a vector of discrete probabilities so that p[x] is the probability of x.
## - X: The vector of x where P is supported
## - f: a real-valued function
## Returns
## - The expectation of f under p, $E_P[f]$
def discrete_expectation(P, X,  f):
    expectation = 0
    ## fill in
    return expectation

def example_function(x):
    return 4*x - 10

# the number of possible values
n_values = 4
# generate a random vector that sums to 1
#P = np.random.dirichlet(np.ones(n_values))
P = [0.1, 0.2, 0.3, 0.4]; 
print(discrete_expectation(P, np.arange(n_values), example_function))
