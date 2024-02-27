# Skeleton code

# Assume the number of models is equal to n=len(prior).  The argument
# P is an n-by-m array, where m is the number of possible
# predictions so that P[i][j] is the probability the i-th model assignsm to the j-th outcome. The outcome is a single number in 1, m.
import numpy as np
import random

## Calculate the posterior given a prior belief, a set of predictions, an outcome
## - prior: belief vector so that prior[i] is the probabiltiy of mdoel i being correct
## - P: P[i][j] is the probability the i-th model assignsm to the j-th outcome
## - outcome: actual outcome
def get_posterior(prior, P, outcome):
    n_models = len(prior)
    posterior = np.zeros(n_models)
    for m in range(n_models):
        posterior[m] = P[i][j] * prior[i] # only the nominator
    posterior /= get_marginal_prediction(prior, P, outcome)
    return posterior


## Get the probability of the specific outcome given your current
## - belief: vector so that belief[i] is the probability of mdoel i being correct
## - P: P[i,j] is the probability the i-th model assigns to the j-th outcome
## - outcome: actual outcome
def get_marginal_prediction(belief, P, outcome):
    n_models = len(belief)
    outcome_probability = sum([P[i][outcome] * prior[i] for i in range(n_models)])
    return outcome_probability

## In this function, U[action,outcome] should be the utility of the action/outcome pair
## Now I calculate \sum_{a,x} P(x) U(a,x)
## Where P(x) is the marginal probability
def get_expected_utility(belief, P, action, U):
    n_models = len(belief)
    n_outcomes = np.shape(P)[1]
    EU = 0
    for k in range(n_outcomes):
        P_outcome = get_marginal_prediction(belief, P, k)
        EU += U[action, k] * P_outcome
    return EU

## In this function, U[action,outcome] should be the utility of the action/outcome pair, using MAP inference
def get_MAP_utility(belief, P, action, U):
    n_models = len(belief)
    n_outcomes = np.shape(P)[1]
    EU = 0
    # get MAP model
    m_MAP = np.argmax(belief)
    for k in range(n_outcomes):
        P_outcome = P[m_MAP][k]
        EU += U[action, k] * P_outcome
    return EU

## Here you should return the action maximising expected utility
## Here we are using the Bayesian marginal prediction
def get_best_action(belief, P, U):
    n_models = len(belief)
    n_actions = np.shape(U)[0]
    V = np.zeros(n_actions)
    for a in range(n_actions):
        V[a] = get_expected_utility(belief, P, a, U)
    return np.argmax(V)
    
## Here you should return the action maximising expected utility
## Here we are using the MAP model
def get_best_action_MAP(belief, P, U):
    n_models = len(belief)
    n_actions = np.shape(U)[0]



T = 4 # number of time steps
n_models = 3 # number of models

# build predictions for each station of rain probability
prediction = np.matrix('0.1 0.1 0.3 0.4; 0.4 0.1 0.6 0.7; 0.7 0.8 0.9 0.99')


n_outcomes = 2 # 0 = no rain, 1 = rain

## we use this matrix to fill in the predictions of stations
P = np.zeros([n_models, n_outcomes])
belief = np.ones(n_models) / n_models;
rain = [0, 0, 1, 0];

print("a x U")
print("-----")
for t in range(T):
    # utility to loop to fill in predictions for that day
    for model in range(n_models):
        P[model,1] = prediction[model,t] # the table predictions give rain probabilities
        P[model,0] = 1.0 - prediction[model,t] # so no-rain probability is 1 - that.
    probability_of_rain = get_marginal_prediction(belief, P, 1)
    #print(probability_of_rain)
    U  = np.matrix('1 -10; 0 0')
    action = get_best_action(belief, P, U)
    MAP_action = get_best_action_MAP(belief, P, U)
    
    print(action, MAP_action, rain[t], U[action, rain[t]], U[MAP_action, rain[t]])
    belief = get_posterior(belief, P, rain[t])
    print(belief)

                
    
