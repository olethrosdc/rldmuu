from abc import ABC
import gymnasium as gym

from gymnasium import spaces

import matplotlib.pyplot as plt
import numpy as np


## Here bandit problems are sampled from a Beta distribution
class BetaBandits(gym.Env):
    def __init__(self,
                 bandits=10,
                 initial_alpha=1,
                 initial_beta=1,
                 seed=None,
                 ):

        self.seed = seed
        self.random = np.random.default_rng(seed)
        self.r_dist = np.zeros(bandits)

        for i in range(bandits):
            self.r_dist[i] = self.random.beta(initial_alpha, initial_beta)

        self.n_bandits = bandits
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)

        # Actual best arm (unknown to the algorithm)
        self.best_arm_param = np.max(self.r_dist)

    def step(self, action):
        assert self.action_space.contains(action), f"Received wrong action {action}"
        done = True
        reward = self.random.binomial(1, self.r_dist[action])
        return 0, reward, done, done, {}

    def reset(self):
        self.random = np.random.default_rng(self.seed)
        return 0, {}


class BanditAlgorithm(ABC):
    """
    Abstract class
    Do not modify
    """
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def act(self) -> int:
        """
        Choose an arm/action
        """
        pass

    def update(self, action, reward):
        """
        Update our strategy
        """
        pass


class RandomSampling(BanditAlgorithm):

    def act(self) -> int:
        return np.random.choice(self.n_actions)


class Greedy(BanditAlgorithm):
    def __init__(self, n_actions):
        super().__init__(n_actions)
        self.alpha = 1 + np.zeros(n_actions)
        self.beta = 0.1 + np.zeros(n_actions)
        self.means = self.alpha / (self.alpha + self.beta)

        #sums = 1 + np.zeros(n_actions)
        #counts = 1 + np.zeros(n_actions)
        #means = sums / (counts)
    def act(self):
        return np.argmax(self.means)


    def update(self, action, reward):
        self.alpha[action] += reward
        self.beta[action] += (1 - reward)
        self.means[action] = self.alpha[action] / (self.alpha[action] + self.beta[action])
        #sums[action] += reward
        #counts[action] += 1
        pass


class EpsilonGreedy(BanditAlgorithm):
    def __init__(self, n_actions):
        super().__init__(n_actions)
        self.alpha = 1 + np.zeros(n_actions)
        self.beta = 1 + np.zeros(n_actions)
        self.means = self.alpha / (self.alpha + self.beta)
        self.epsilon = 0.01
        self.n_steps = 1
        #sums = 1 + np.zeros(n_actions)
        #counts = 1 + np.zeros(n_actions)
        #means = sums / (counts)
    def act(self):
        self.epsilon = 100/(100 +self.n_steps)
        if (np.random.uniform() < self.epsilon):
            return np.random.choice(self.n_actions)
        return np.argmax(self.means)


    def update(self, action, reward):
        self.n_steps += 1
        self.alpha[action] += reward
        self.beta[action] += (1 - reward)
        self.means[action] = self.alpha[action] / (self.alpha[action] + self.beta[action])
        #sums[action] += reward
        #counts[action] += 1
        pass


class UCB(BanditAlgorithm):
    def __init__(self, n_actions):
        super().__init__(n_actions)

        ...

    def act(self):
        pass

    def update(self, action, reward):
        pass


class ThompsonSampling(BanditAlgorithm):
    def __init__(self, n_actions):
        super().__init__(n_actions)

        ...

    def act(self):
        pass

    def update(self, action, reward):
        pass



# Smooth down the values, to better visualize improvement
def smooth_curve(x, K):
    T = x.shape[0]
    smoothed_x = np.empty_like(x)
    for t in range(0, T):
        min_idx = np.maximum(0, t - K)
        max_idx = np.minimum(T, t + K)
        smoothed_x[t, :] = np.mean(x[min_idx: max_idx], axis=0)

    return smoothed_x


if __name__ == '__main__':

    # --------------------------- Experiments ---------------------------
    # You can try changing these
    # More actions requires more learning steps
    n_actions = 100

    n_experiments = 30
    T = 2000
    environments = []

    # Instantiate some bandit problems
    for experiment_id in range(n_experiments):
        environments.append(BetaBandits(n_actions, 1, 1, seed=experiment_id))

    # The algorithms we want to benchmark
    algs = [RandomSampling, Greedy, EpsilonGreedy]

    n_algs = len(algs)
    reward_t = np.zeros((T, n_algs), dtype=np.float32)
    regret_t = np.zeros((T, n_algs), dtype=np.float32)
    mean_best = 0
    total_reward = np.zeros(n_algs)
    for experiment in range(n_experiments):
        env = environments[experiment]
        mean_best += env.best_arm_param
        print("Running experiment N°", experiment)
        for alg_index, Alg in enumerate(algs):
            np.random.seed(experiment)
            alg = Alg(n_actions)
            run_reward = 0
            env.reset()
            for i_episode in range(T):

                # we choose an arm (action)
                action = alg.act()

                # Observe the reward for choosing the action
                _, reward, _, _, _ = env.step(action)  # play the action in the environment

                # learn
                alg.update(action, reward)

                run_reward += reward
                reward_t[i_episode, alg_index] += reward
                regret_t[i_episode, alg_index] += env.best_arm_param - reward

            total_reward[alg_index] += run_reward

    total_reward /= n_experiments
    reward_t /= n_experiments
    regret_t /= n_experiments
    mean_best /= n_experiments
    cumulative_regret = np.cumsum(regret_t, axis=0)

    smoothing = 20
    plt.plot(smooth_curve(reward_t, smoothing))
    plt.plot(np.ones(T)*mean_best)
    plt.legend([c.__name__ for c in algs])
    plt.ylabel("Average reward")
    plt.xlabel("Timesteps")
    plt.savefig("benchmark_reward.pdf")
    plt.clf()

    plt.plot(smooth_curve(cumulative_regret, smoothing))
    plt.legend([c.__name__ for c in algs])
    plt.ylabel("Cumulative regret")
    plt.xlabel("Timesteps")
    plt.title("Total algorithm regret")
    plt.savefig("benchmark_regret.pdf")
