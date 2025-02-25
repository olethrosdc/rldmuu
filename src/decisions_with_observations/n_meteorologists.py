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
    pass


## Get the probability of the specific outcome given your current
## - belief: vector so that belief[i] is the probability of mdoel i being correct
## - P: P[i,j] is the probability the i-th model assigns to the j-th outcome
## - outcome: actual outcome
def get_marginal_prediction(belief, P, outcome):
    pass

## In this function, U[action,outcome] should be the utility of the action/outcome pair
## Now I calculate \sum_{a,x} P(x) U(a,x)
## Where P(x) is the marginal probability
def get_expected_utility(belief, P, action, U):
    pass

## In this function, U[action,outcome] should be the utility of the action/outcome pair, using MAP inference
def get_MAP_utility(belief, P, action, U):
    pass

## Here you should return the action maximising expected utility
## Here we are using the Bayesian marginal prediction
def get_best_action(belief, P, U):
    pass
    
## Here you should return the action maximising expected utility
## Here we are using the MAP model
def get_best_action_MAP(belief, P, U):
    pass



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

                
    
