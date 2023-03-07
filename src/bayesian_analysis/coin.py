# Skeleton code

## Here we want to simply predict whether a coin is falling heads or tails
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class BetaConjugatePrior:
    ## Initialise with the beta parameters
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    ## Update the belief when you see a new observation
    def update(self, observation):
        if observation == 0:
            self.alpha += 1
        else:
            self.beta += 1
        return

    ## get the probability of a new observation $P_\xi(x)$
    def get_marginal_probability(self, observation):
        p = self.alpha / (self.alpha + self.beta)
        if observation == 1:
            p = 1 - p
        return p

    ## Sample a Bernoulli parameter $\omega \sim Beta(\alpha, \beta)$
    def sample_parameter(self):
        return np.random.beta(self.alpha, self.beta)


T = 20  # number of time steps
# an array of tails (0) and heads (1)
true_bias = 0.6
previous_belief = BetaConjugatePrior(1, 1)
belief = BetaConjugatePrior(1, 1)

x = np.linspace(0, 1, 20)


def visualize_distribution():
    plt.plot(x, stats.beta.pdf(x, belief.alpha, belief.beta))
    plt.plot(x, stats.beta.pdf(x, previous_belief.alpha, previous_belief.beta), linestyle="--")
    plt.legend(["Current belief model", "Previous belief model"])
    plt.show()


visualize_distribution()

for t in range(T):
    # sample from the n=1 binomial (i.e. Bernoulli) distribution
    observation = np.random.binomial(1, true_bias)
    # get the probability of the observation according to the current belief
    probability_of_observation = belief.get_marginal_probability(observation)
    ## update the belief
    belief.update(observation)

    print("Number of successes:", belief.alpha, "Number of fails", belief.beta)

    if (t + 1) % 1 == 0:
        visualize_distribution()

    previous_belief.update(observation)


