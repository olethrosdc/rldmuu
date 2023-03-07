## -*-

import numpy as np
n_models = 10
horizon = 1000
rain_probabilities = np.random.uniform(size = horizon)
P_model = np.zeros([n_models, horizon])

for t in range(n_models):
    alpha = 1 - (t + 0.5) / n_models
    P_model[t] = alpha * np.random.uniform(size = horizon) + (1 - alpha) * rain_probabilities

print(P_model)
print(rain_probabilities)
eta = 1

def get_expected_utility(a, p, u):
    return u[a,0] * (1 - p) + u[a,1] * p

def get_best_action(p, u):
    v = np.zeros(2)
    for a in range(2):
        v[a] = get_expected_utility(a, p, u)
    return np.argmax(v)

util=np.matrix('1 -1; 0 1')
belief_Bayes = np.ones(n_models) / n_models
belief_util = np.ones(n_models) / n_models
bBayes = np.zeros([horizon, n_models])
bUtil = np.zeros([horizon, n_models])
for t in range(horizon):
    y_t = np.random.binomial(1, rain_probabilities[t])

    ## Update Bayes belief
    if (y_t==0):
        likelihood = 1 - P_model[:,t]
    else:
        likelihood = P_model[:,t]

    print(likelihood.shape)
    belief_Bayes *= likelihood
    belief_Bayes /= sum(belief_Bayes)

    ## Update utility belief
    u_t = np.zeros(n_models)
    for m in range(n_models):
        p_t = P_model[m,t]
        a_tm = get_best_action(p_t, util)
        u_t[m] = util[a_tm, y_t]
    belief_util *= np.exp(u_t)
    belief_util /= sum(belief_util)

    print("Round ", t)
    print(belief_Bayes)
    print(belief_util)
    bBayes[t] = belief_Bayes
    bUtil[t] = belief_util

import matplotlib.pyplot as plt
plt.plot(bBayes[:,n_models - 1])
plt.plot(bUtil[:,n_models - 1])
plt.legend(["Bayes", "Util"])
plt.savefig("ewa.pdf")

    
