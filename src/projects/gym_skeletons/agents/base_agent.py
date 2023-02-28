from abc import ABC, abstractmethod

import gymnasium


class BaseAgent(ABC):

    def __init__(self,
                 env: gymnasium.Env
                 ):

        self.env = env

    @abstractmethod
    def make_decision(self, observation, explore=False) -> int:
        """
        :param observation: observation we base ourselves on to make the action
        :param explore: Are we exploring ?
        :return: the action index
        """

        # By default, randomly choose an action
        return self.env.action_space.sample()

    @abstractmethod
    def learn(self, state, action, reward, next_state) -> dict:
        """
        :param state: the state we were at when choosing the action
        :param action: the action chosen
        :param reward: the reward received for choosing the action
        :param next_state: the state reached after making the action
        :return: a dictionary with some info regarding learning (training loss, other metrics, etc)
        """
        # Update our strategy based on the observations we just made.
        return {}
