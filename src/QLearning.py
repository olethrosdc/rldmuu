import numpy as np

class QLearning:
    def __init__(self, n_actions, n_states, discount=0.9, alpha = 0.01):
        self.n_actions = n_actions
        self.n_states = n_states
        self.Q = np.zeros([n_states, n_actions])
        self.discount = discount
        self.alpha = 0.01
    def act(self):
        if (np.random.uniform() < self.epsilon):
            return np.random.choice(self.n_actions)
        return np.argmax(Q[self.state, :])
    
    def update(self, action, reward, state):
        self.Q[self.state, action] += alpha * np.max(reward + self.discount * Q[state, :] - Q[self.state, action])
        self.state = state

    def reset(self, state):
        self.state = state
