import os
from time import sleep

import numpy as np
from gymnasium import spaces
from numpy import ndarray

from .base_env import BaseEnv


class GridWorldEnvSparce(BaseEnv):

    """
    Example environment

    The agent starts at some random integer coordinates (x, y).
    Its goal is to reach the goal coordinates (x_g,y_g) by navigating the space,
    by moving either right, up, left or down.
    """

    def __init__(self, render=False, size=5):
        # Initialize the base class
        super().__init__(render)

        # The observation contains the 4 required coordinates (x, y, x_g, y_g)
        self.observation_space = spaces.Box(0, size - 1, shape=(4,), dtype=int)
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        # Environment configuration
        self.size = size

        # Environment state attributes
        self._agent_location = None
        self._target_location = None
        self.reached_goal = False
        self.past_agent_location = None

        """
        Rendering is necessary if we want to:
        - debug the environment
        - visualize the agent's behavior
        """
        # Instantiate an array useful for rendering
        self.console_rendered = np.full(
            (size, size), dtype=str, fill_value="□"
        )

    def get_observation(self) -> ndarray:
        """
        :return: (one dimension array) the observation we want to feed to the agent,
        based on the current state of the environment.
        """
        # We return the array containing (x, y, x_g, y_g)
        return np.concatenate([self._agent_location, self._target_location])

    def reset(self) -> ndarray:

        self.reached_goal = False
        # Choose the agent's location uniformly at random
        self._agent_location = np.random.randint(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.random.randint(0, self.size, size=2, dtype=int)

        # get the initial observation
        observation = self.get_observation()
        self.past_agent_location = self._agent_location.copy()


        # return the observation and some additional information we could want to pass
        return observation

    def render(self):
        """
        Prints to the console the gridworld
        """

        self.console_rendered[:] = "□"
        if np.array_equal(self._agent_location, self._target_location):
            # The goal state has been reached
            self.console_rendered[self._agent_location[0], self._agent_location[1]] = "O"
        else:
            # Our current location
            self.console_rendered[self._agent_location[0], self._agent_location[1]] = "▣"
            # The goal's location
            self.console_rendered[self._target_location[0], self._target_location[1]] = "G"

        # Print to the console the state of the environment
        os.system('clear')
        print("\n")
        for line in self.console_rendered:
            print("\n\t" + "   ".join(line))

        # Sleep for visibility
        sleep(0.33)

    def update(self, action) -> bool:
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the allowed space
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        return terminated

    def compute_reward(self, action) -> float:
        # Binary sparce rewards.
        # If we are terminated, then this means we reached the goal.
        return int(self.reached_goal)

    def step(self, action):

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        terminated = self.update(action)
        self.reached_goal = terminated
        observation = self.get_observation()
        reward = self.compute_reward(action)

        self.past_agent_location = self._agent_location.copy()

        return (
            observation,  # updated observation.
            reward,  # reward for taking the action.
            terminated  # is the episode finished ?
        )

    def close(self):
        pass


class GridWorldEnv(GridWorldEnvSparce):
    """
    GridWorld environment with non-sparce rewards.
    """

    @staticmethod
    def manhattan_distance(pos1, pos2):
        return np.sum(np.abs(pos2-pos1))

    def compute_reward(self, action) -> float:
        # If we get closer to the goal, get a reward, otherwise receive a penalty
        return (
                self.manhattan_distance(self._agent_location, self._target_location)
                - self.manhattan_distance(self.past_agent_location, self._target_location)
        )

GridWorldEnvSparceGym = GridWorldEnvSparce.to_gym_env
GridWorldEnvGym = GridWorldEnv.to_gym_env
