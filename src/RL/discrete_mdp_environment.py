import numpy as np

from MDP import DiscreteMDP

## Import the GYM API
import gym

## This is a bandit environment
from gym import spaces
from gym.utils import seeding

## Here bandit problems are sampled from a Beta distribution
class OAI_DiscreteMDP(gym.Env):
  def __init__(self, states=10, actions=2):
    self.n_states = states
    self.n_actions = actions
    self.action_space = spaces.Discrete(self.n_actions)
    self.observation_space = spaces.Discrete(self.n_states)
    self.mdp = DiscreteMDP(states, actions)
    self.state = 0
    self._seed()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    assert self.action_space.contains(action)
    p_sa = self.mdp.get_transition_probabilities(self.state, action)
    reward = self.mdp.get_reward(self.state, action)
    self.state = np.random.choice(range(self.n_states), p=p_sa)

    done = False
    
    return self.state, reward, done, {}

  def reset(self):
    return 0

  def render(self, mode='human', close=False):
    pass
  
class RandomAlgorithm:
    def __init__(self, n_actions):
        self.n_actions = n_actions
    def act(self):
        return np.random.randint(self.n_actions)
    ## Stochastic update: mu = mu + alpha * z
    ## z = r - mu
    def update(self, action, reward):
      pass

n_states = 4
n_actions = 2
n_experiments = 10
T = 100
environments = []
for experiment in range(n_experiments):
  environments.append(OAI_DiscreteMDP(n_states, n_actions))

algs = []
algs.append(RandomAlgorithm)
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
        action = alg.act() # function for taking an action
        observation, reward, done, info = env.step(action) # action take in the environment
        alg.update(action, reward, observation)
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
plt.plot(moving_average(reward_t, 100))
plt.legend(["Greedy", "Stochastic"])
plt.savefig("stochastic.pdf")
#  plt.show()
  
 


