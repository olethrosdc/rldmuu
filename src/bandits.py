## Import the GYM API
import gym
## This is a bandit environment
import gym_bandits

class AverageBanditAlgorithm:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.total_reward = np.zeros(n_actions)
        self.n_pulls = np.zeros(n_actions)
    def act(self):
        ## FILL IN
        return action


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

