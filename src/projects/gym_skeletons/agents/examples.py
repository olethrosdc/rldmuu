import gymnasium
import numpy as np

from .base_agent import BaseAgent


class Qlearning(BaseAgent):
    """
    Q-learning algorithm
    No exploration
    """

    def __init__(
            self,

            env: gymnasium.Env,

            discount=0.99,
            learning_rate=0.01,
                 ):
        super().__init__(env)

        self.Q = {}
        self.discount = discount
        self.learning_rate = learning_rate

    def make_decision(self, observation, explore=True):
        return self.best_action(tuple(observation))

    def best_action(self, hashable_state):
        return max(
            range(self.env.action_space.n),
            key=lambda next_a: 0. if (hashable_state, next_a) not in self.Q else self.Q[(hashable_state, next_a)]
        )

    def learn(self, state, action, reward, next_state):
        """
        updates the Q table with Q[s, a] := Q[s, a] + α[r + γ maxa' Q(s', a') - Q(s, a)]
        """

        hashable_state = tuple(state)
        hashable_next_state = tuple(next_state)

        s_a = hashable_state, action
        if s_a not in self.Q:
            self.Q[s_a] = 0.

        next_state_action = hashable_next_state, self.best_action(hashable_next_state)

        if next_state_action not in self.Q:
            self.Q[next_state_action] = 0.

        bellman = self.Q[s_a] + self.learning_rate \
                  * (reward + self.discount * self.Q[next_state_action] - self.Q[s_a])

        loss = bellman - self.Q[s_a]
        self.Q[s_a] = bellman

        return loss


class EpsilonGreedyQlearning(Qlearning):
    """
    Explores greedily
    """

    def __init__(
            self,

            env: gymnasium.Env,

            discount=0.99,
            learning_rate=0.01,
            eps_greedy_0=1.,
            eps_decay=0.9999,
            eps_final=1e-2,

                 ):
        super().__init__(
            env,
            discount=discount,
            learning_rate=learning_rate
        )

        self.eps = eps_greedy_0
        self.eps_decay = eps_decay
        self.eps_final = eps_final

    def make_decision(self, observation, explore=True):
        if explore and np.random.random() < self.eps:
            return self.env.action_space.sample()
        else:
            return super().make_decision(observation, explore)


    def learn(self, state, action, reward, next_state):
        """
        updates the Q table with Q[s, a] := Q[s, a] + α[r + γ maxa' Q(s', a') - Q(s, a)]
        """

        loss = super().learn(state, action, reward, next_state)

        self.eps = np.clip(self.eps*self.eps_decay, self.eps_final, 1)

        return loss


class SoftQlearning(Qlearning):
    """
    Softmax exploration
    """

    def __init__(
            self,

            env: gymnasium.Env,

            discount=0.99,
            learning_rate=0.01,
            eta0=1000.,
            eta_decay=0.999,
            eta_final=1e-2,

                 ):
        super().__init__(
            env,
            discount=discount,
            learning_rate=learning_rate
        )

        self.eta = eta0
        self.eta_decay = eta_decay
        self.eta_final = eta_final

    def make_decision(self, observation, explore=True):
        if explore:
            hashable_state = tuple(observation)
            q_values = [
                0. if (hashable_state, a) not in self.Q else self.Q[(hashable_state, a)] / self.eta
                for a in range(self.env.action_space.n)
            ]
            exp_q = np.exp(q_values)

            p = exp_q / np.sum(exp_q)

            # sample over softmax distribution

            return np.random.choice(self.env.action_space.n, p=p)

        else:
            return super().make_decision(observation, explore)

    def learn(self, state, action, reward, next_state):
        """
        updates the Q table with Q[s, a] := Q[s, a] + α[r + γ maxa' Q(s', a') - Q(s, a)]
        """

        loss = super().learn(state, action, reward, next_state)
        self.eta = np.clip(self.eta*self.eta_decay, self.eta_final, 1000)

        return loss



