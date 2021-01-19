# Skeleton code

## Here we want to simply predict whether a coin is falling heads or tails
import numpy as np
import scipy.stats as stats

class BetaConjugatePrior:
    ## Initialise with the beta parameters
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    ## Update the belief when you see a new observation
    def update(self, observation):
        ## FILL
        return
    ## get the probability of a new observation $P_\xi(x)$
    def get_marginal_probability(self, observation):
       ## FILL
        return probability
    ## Sample a Bernoulli parameter $\omega \sim Beta(\alpha, \beta)$
    def sample_parameter(self):
       ## FILL
        return omega

T = 100 # number of time steps

# an array of tails (0) and heads (1)
outcomes = np.array('0 1 0 1 0 1 0 0 1 0 1')
true_bias = 0.6
belief = BetaConjugatePrior(1,1)

print("-----")
for t in range(T):
    # sample from the n=1 binomial (i.e. Bernoulli) distribution
    observation = np.random.binomial(1, true_bias)
    # get the probability of the observation according to the current belief
    probability_of_observation = belief.get_marginal_probability(observation)
    ## update the belief
    belief.update(observation)

    print(belief.get_marginal_probability(1))

                
    
