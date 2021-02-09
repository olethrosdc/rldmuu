## Import the GYM API
import gym
## This is a bandit environment
import gym_bandits


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
        return argmax(total_reward./n_pulls)
    def update(self, action, reward):
        self.total_reward[action] += reward
        self.n_pulls[action] += 1

class StochasticBanditAlgorithm:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.mean = np.ones(n_actions)
        self.alpha = 0.01 * np.ones(n_actions)
    def act(self):
        return argmax(mean)
    ## Stochastic update: mu = mu + alpha * z
    ## z = r - mu
    def update(self, action, reward):
        self.mean[alpha] += self.alpha[action] * (reward - mu[alpha])


class BayesOptimalBernoulliBandit:
    def __init__(self, n_actions, horizon):
        self.n_actions = n_actions
        self.horizon = horizon # maximum steps to look ahead
        ## Beta distribution parameters
        self.alpha = np.ones(n_actions)
        self.beta = np.ones(n_actions)

    def recurse_utility(self, belief):
        # careful here not to overwrite the actual self.alpha and self.beta when recursing
        return max_utility
    def calculate_utility(self, action):
        # Calculate the expected return of the action by marginalising over rewards and next beliefs. This can be done by recursion
    def act(self):
        ## Fill in
        U = np.zeros(n_actions)
        for i in range(self.n_actions):
            U[i] = calculate_utility(i)
        return argmax(U)
    def update(self, action, reward):
        self.alpha[action] += reward
        self.beta[action] += (1 - reward)
    

  
for horizon in range(4):
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
      alg = Alg(n_actions, horizon)
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
  print("H:", horizon, "U:", total_reward)
  plt.clf()
  plt.plot(moving_average(reward_t, 10))
  plt.legend(["Greedy", "Stochastic"])
  plt.savefig("horizon" + str(horizon) + ".pdf")
#  plt.show()
  
 


