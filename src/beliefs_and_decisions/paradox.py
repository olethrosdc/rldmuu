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


def quadratic_utility(money:
     ## fill in
return money                    


# There is not always a closed-form solution for the expected utility,
# hence we are doing a numerical approximation.  In your function, you
# should calcualte the expected utility of playing, depending on the
# payment and the starting capital.
#
# For the standard scenario with linear utilities, the approximation
# will always strongly underestimate the utility.
def expected_utility(starting_capital, payment, utility_function):
    EU = 0
    p_stop = 0.5 # the probability of stopping at any step
    gain = 2 # the gain in money for every step that passes
    # here, you should calculate the total expected utility up to some point
    for stopping in range(1000):
    # fill in
    return EU

utility_function = log_utility
starting_capital = 10

for payment in range(starting_capital):
    no_play_util = utility_function(payment)
    print(play_exp_util, no_play_util)
    if (play_exp_util < no_play_util):
        print("Not playing if payment larger than ", payment, " out of ", starting_capital, "starting capital")
        break



