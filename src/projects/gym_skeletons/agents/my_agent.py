from rldmuu.src.projects.gym_skeletons.agents.base_agent import BaseAgent


class MyAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)

    def make_decision(self, observation, explore=False) -> int:
        pass

    def learn(self, state, action, reward, next_state) -> dict:
        pass

