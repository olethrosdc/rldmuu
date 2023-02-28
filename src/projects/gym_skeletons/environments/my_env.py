from gymnasium.core import ObsType
from numpy import ndarray

from .base_env import BaseEnv


class MyEnv(BaseEnv):

    def __init__(self, render):
        super().__init__(render)

    def reset(self) -> ndarray:
        pass

    def get_observation(self) -> ndarray:
        pass

    def compute_reward(self, action) -> float:
        pass

    def update(self, action) -> bool:
        pass

    def render(self) -> None:
        pass

    def step(self, action) -> tuple[ObsType, float, bool]:
        pass


"""
to reuse our environment in experiments, we convert our environment to a gym adapted environment with:

>>>> GymEnvClassName = MyEnv.to_gym_env

and add the following to the __init__.py of the environments package:

register(
    id="MyEnv", # Name of our environment
    entry_point="environments.base_env:MyEnvGym", # Essentially: gym_skeletons.environments.file_name:GymEnvClassName
    max_episode_steps=300, # Forces the environment episodes to end once the agent played for max_episode_steps steps
)
"""
MyEnvGym = MyEnv.to_gym_env








