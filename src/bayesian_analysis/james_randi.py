import numpy as np

# James Randi exercise

# Exercise 2

p0 = 2 ** (-16)
belief = [p0, 1 - p0]

n_models = 2  # one model if we believe james is a psychic, another for not believing
n_outcomes = 2  # one outcome for predicting the throw, another for not predicting
P = np.zeros((n_models,n_outcomes))

# Event of guessing correctly
B = 0
# event of James being a psychic
A = 0

# Probability of guessing correctly, knowing he's a psychic
P[A, B] = 1
# Probability of not guessing correctly, knowing he's a psychic (it's 0)
P[A, ~B] = 1 - P[A, B]  # (~B == 1, bitwise not)

# Probability of guessing correctly or not, knowing he's not a psychic
P[~A, B] = 0.5
P[~A, ~B] = 1 - P[~A, B]


# The marginal probability, on the outcome
def get_marginal(belief, P, outcome):
    n_models = len(belief)
    outcome_probability = sum([belief[i] * P[i][outcome] for i in range(n_models)])
    return outcome_probability


def get_posterior(prior, P, outcome):
    n_models = len(prior)
    posterior = np.zeros(n_models)
    for m in range(n_models):
        posterior[m] = prior[m] * P[m][outcome]
    posterior /= sum(posterior)
    return posterior

P_Bk = []
# Probability of guessing right, with no data
P_Bk.append(get_marginal(belief, P, B))

for i in range(19):
    # We see James guessing right, we update our belief regarding his psychic powers
    belief = get_posterior(belief, P, B)

    # Probability that James guesses right, after observing that he guessed right on the i+1 previous rounds
    P_Bk.append(get_marginal(belief, P, B))

# We win 100CU if James Randi does not guess correctly 4 times in a row.
# The expected utility is 100 * time the probability that he fails.
# We have an initial belief regarding his psychic power, of probability p0.
# This has an impact over the expected utility. We want to not bet anything, if he trully is a psychic !
EU_1 = (1 - 2*np.prod([P_Bk[i] for i in range(0, 4)])) * 100 # -> 93.75
# We already witnessed James guessing 4 times in a row
EU_2 = (1 - 2*np.prod([P_Bk[i] for i in range(4, 8)])) * 100
# 8 times in a row ...
EU_3 = (1 - 2*np.prod([P_Bk[i] for i in range(8, 12)])) * 100
EU_4 = (1 - 2*np.prod([P_Bk[i] for i in range(12, 16)])) * 100
EU_5 = (1 - 2*np.prod([P_Bk[i] for i in range(16, 20)])) * 100 # -> 46.87

print(P_Bk)
print(EU_1, EU_2, EU_3, EU_4, EU_5)


# ------------------------------------------------------------------------------------------------
# Exercice 3.3
# James Randi can predict correctly with a probability \theta in [0,1].
# The difference now is that we don't know his prediction accuracy,
# Previously, we knew that \theta==1, now we need to also have a belief over \theta
# We model our belief \xi(\theta)) with a Beta distribution Beta(\alpha,\beta) = Beta(2,1).
# Here, \alpha models the amount of correct guesses, and \beta the amount of wrong guesses.
# Note : E[\xi(\theta)] = \alpha / (\alpha + \beta)
# As we see James predicting outcomes, we need to update our belief regarding his psychic power,
# and his psychic accuracy .


# The Beta distribution is useful to model our belief regarding James quality as a psychic
# cf. course book pp.65-66
class BetaConjugatePrior:
    # Initialise with the beta parameters
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    # Update the belief when you see a new observation
    def update(self, observation):
        # Each time we observe something new, the posterior distribution still correspond
        # to a Beta distribution, but with updated parameters.

        # TODO

        pass

    # get the probability of a new observation $P_\xi(x)$
    def get_marginal_probability(self, observation):
        probability = 0

        # TODO

        return probability

    ## Sample a Bernoulli parameter $\omega \sim Beta(\alpha, \beta)$
    def sample_parameter(self):
        return np.random.beta(self.alpha, self.beta)

    # Get the expected value
    def get_expected(self):
        pass

# James' true probability to predict a toss (we can play around with it)
# This is what we want to model and learn to maximize our gains
true_theta = 0.75

alpha = 2
beta = 1

# model for our belief on James' predicting ability
prediction_chance_belief = BetaConjugatePrior(alpha, beta)
# initially, we still believe that James is a psychic with P(a) = 2^-16
psychic_belief = [p0, 1-p0]

keep_playing = True
money = 0
round = 0

# Probability we believe that James correctly guesses if he is a psychic:
expected_theta = ...

P[A, B] = ...
P[A, ~B] = 1 - P[A, B]

while keep_playing:

    # We let James guess
    #  He correctly guesses with a probability true_theta
    # correct_guess = 0 means that James correctly guessed
    correct_guess = ...

    # We update our beliefs, seeing James guessing correctly or not.

    # TODO

    # if he guessed correctly, we lose 100, otherwise, we win 100
    money = ...

    # Get our new estimated parameter
    expected_theta = ...

    # We update our model
    P[A, B] = ...
    P[A, ~B] = 1 - P[A, B]

    # What is the probability that James correctly predicts next round's toss, with our updated beliefs ?
    next_guessing_probability = ...

    print(money, next_guessing_probability, expected_theta, belief)

    # Find a strategy that tells us whether we want to keep playing or not ?
    keep_playing = ...

    round += 1