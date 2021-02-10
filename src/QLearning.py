import numpy as np

class QLearning:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.total_reward = np.ones(n_actions)
        self.n_pulls = np.ones(n_actions)
    def act(self):
        return np.argmax(self.total_reward/self.n_pulls)
    def update(self, action, reward, observation):
        self.total_reward[action] += reward
        self.n_pulls[action] += 1
