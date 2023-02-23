#!/usr/bin/env python
import numpy as np
# define the utility function

## Here we define a logarithmic utility function when money is > 0, otherwise
## we return the actual amount of money
def log_utility(money):
    if (money > 0):
        return np.log(money + 1)
    else:
        return money

def linear_utility(money):
    return money


# there is not always a closed-form solution for the expected utility,
# hence we are doing a numerical approximation
def expected_utility(starting_capital, payment, utility_function):
    EU = 0
    p_stop = 0.5
    gain = 2
    for stopping in range(100):
    # fill in
    return EU

utility_function = log_utility
starting_capital = 100

for payment in range(starting_capital):
    no_play_util = utility_function(payment)
    print(play_exp_util, no_play_util)
    if (play_exp_util < no_play_util):
        print("Not playing if payment larger than ", payment, " out of ", starting_capital, "starting capital")
        break



