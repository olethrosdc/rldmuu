## Import the GYM API
import gym
## This is a bandit environment
import gym_bandits
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt

import numpy as np


def moving_average(x, K):
  T = x.shape[0]
  n = x.shape[1]
  m = int(np.ceil(T / K))
  y = np.zeros([m, n])
  for alg in range(n):
      for t in range(m):
        y[t,alg] = np.mean(x[t*K:(t+1)*K, alg])
  return y


## Here bandit problems are sampled from a Beta distribution
class BetaBandits(gym.Env):
  def __init__(self, bandits=10, alpha=1, beta=1):
    self.r_dist = np.zeros(bandits)
    for i in range(bandits):
      self.r_dist[i] = np.random.beta(alpha, beta)
    self.n_bandits = bandits
    self.action_space = spaces.Discrete(self.n_bandits)
    self.observation_space = spaces.Discrete(1)

    self._seed()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    assert self.action_space.contains(action)
    done = True
    reward = np.random.binomial(1, self.r_dist[action])
    return 0, reward, done, {}

  def reset(self):
    return 0

  def render(self, mode='human', close=False):
    pass



class AverageBanditAlgorithm:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.total_reward = np.ones(n_actions)
        self.n_pulls = np.ones(n_actions)
    def act(self):
        return np.argmax(self.total_reward/self.n_pulls)
    def update(self, action, reward):
        self.total_reward[action] += reward
        self.n_pulls[action] += 1

class StochasticBanditAlgorithm:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.mean = np.ones(n_actions)
        self.alpha = 0.01 * np.ones(n_actions)
    def act(self):
        return np.argmax(self.mean)
    ## Stochastic update: mu = mu + alpha * z
    ## z = r - mu
    def update(self, action, reward):
        self.mean[action] += self.alpha[action] * (reward - self.mean[action])




n_actions = 2
n_experiments = 100
T = 10000
environments = []
for experiment in range(n_experiments):
  environments.append(BetaBandits(n_actions, 1, 1))

algs = []
algs.append(AverageBanditAlgorithm)
algs.append(StochasticBanditAlgorithm)
n_algs = len(algs)
reward_t = np.zeros([T, n_algs])
total_reward = np.zeros([n_algs])
for experiment in range(n_experiments):
  env = environments[experiment]
  env.reset()
  alg_index = 0
  for Alg in algs:
    alg = Alg(n_actions)
    run_reward = 0
    for i_episode in range(T):
      observation = env.reset()
      for t in range(100):
        env.render()
        action = alg.act()
        observation, reward, done, info = env.step(action)
        alg.update(action, reward)
        run_reward += reward
        reward_t[i_episode, alg_index] += reward
        if done:
          #            print("Episode finished after {} timesteps".format(t+1))
          break
    total_reward[alg_index] += run_reward
    alg_index += 1
    env.close()

total_reward /= n_experiments
reward_t /= n_experiments
plt.clf()
plt.plot(moving_average(reward_t, 10))
plt.legend(["Greedy", "Stochastic"])
plt.savefig("stochastic.pdf")
#  plt.show()
  
 


