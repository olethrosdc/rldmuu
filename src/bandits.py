## Import the GYM API
import gym
## This is a bandit environment
import gym_bandits

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
    

# Set up the bandit environment
env = gym.make("BanditTenArmedGaussian-v0")
alg = AverageBanditAlgorithm(env.action_space.n)
env.reset()
total_reward = 0
for i_episode in range(1000):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("A:, R:", action, reward)
        total_reward += reward
        if done:
#            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
print("total reward: ", total_reward)

