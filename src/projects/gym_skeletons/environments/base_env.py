from abc import ABC, abstractmethod
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
from numpy import ndarray


class BaseEnv(ABC):
    """
    Abstract environment class, to be subclassed
    """
    @classmethod
    def to_gym_env(cls, **config):
        return BaseEnvGymWrapper(cls(**config))

    def __init__(self, render=False, **kwargs):
        """
        Define the observation/action spaces there, and eventually some other important values to your environment.
        """
        if not render: self.render = lambda: None

        # The nature of the observation seen by the agent interacting with the environment
        # example:
        # We want an observation space representing our position in a gridworld of size 10x10
        # We thus need two information, our x and y coordinates.
        # *We only will consider discrete observation spaces, hence the int datatype.*
        # >> self.observation_space = spaces.Box(low=0, high=9, shape=(2,), dtype=int)
        # Action 0 is "up", 1 is "right" and so on.
        self.observation_space: spaces.Box = None

        # The possible actions that can be taken by the agent
        # example:
        # We want an action space for "up, right, down, left"
        # >> self.action_space = spaces.Discrete(4)
        # Action 0 is "up", 1 is "right" and so on.
        self.action_space: spaces.Discrete = None

    @abstractmethod
    def reset(self) -> ndarray:
        """
        Resets the environment to an initial state. called at initialization and everytime between episodes
        :return: the initial observation
        """
        assert self.observation_space.dtype == int, "We only want discrete observation spaces"

        # Here, we reset the attributes so that we can start another episode.

        # We return a random observation by default
        return self.get_observation()

    @abstractmethod
    def get_observation(self) -> ndarray:
        """
        :return: (one dimension array) the observation we want to feed to the agent. The observation of the current
        environment state. This is the only data observed by the agent.
        """
        # We return a random observation by default
        return self.observation_space.sample()

    @abstractmethod
    def compute_reward(self, action) -> float:
        """
        Computes the reward when choosing "action".
        This generally depends on the current state we are in, and eventually past states.

        :param action: the chosen action.
        :return: the reward related to the action.
        """

        return 0.

    @abstractmethod
    def update(self, action) -> bool:
        """
        Implements the core mechanisms of the environment.
        - How do we want to update the state of the environment depending on "action" ?

        :param action: the chosen action.
        :return: whether we reached a terminal state or not (the episode is terminated).
        """

        return False

    @abstractmethod
    def render(self) -> None:
        """
        Displays the current state of the environment, for visualization and interpretability
        Could just be a print with a sleep for visibility.
        """

        pass

    @abstractmethod
    def step(self, action) -> tuple[ObsType, float, bool]:
        """
        Updates the environment with the chosen action and returns the corresponding new state and reward.

        :param action: action sent to the environment
        :return: (
            observation,    # the new observation made after updating the environment with the chosen action
            reward,         # returns the reward linked to the transition (past_observation, action, new_observation)
            terminated      # is the episode finished ? if we return True, reset is called next
                  )
        """

        terminated = self.update(action)
        new_observation = self.get_observation()
        reward = self.compute_reward(action)

        return new_observation, reward, terminated


class BaseEnvGymWrapper(gym.Env):
    metadata = {}

    def __init__(self, base_env: BaseEnv):
        super().__init__()
        self._env = base_env
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        observation = self._env.reset()
        self._env.render()

        return observation, {}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done = self._env.step(action)
        self._env.render()

        return obs, reward, done, False, {}

    def render(self) -> None:
        pass


