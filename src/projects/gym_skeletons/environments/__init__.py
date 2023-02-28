from gymnasium.envs.registration import register

register(
     id="GridWorldSparce",  # Name of our environment
     entry_point="gym_skeletons.environments.examples:GridWorldEnvSparceGym",
     # Essentially: gym_skeletons.environments.file_name:GymEnvClassName
     max_episode_steps=300,  # Forces the environment episodes to end once the agent played for max_episode_steps steps,
     # you could consider it to be a timeout
)

register(
     id="GridWorld",
     entry_point="gym_skeletons.environments.examples:GridWorldEnvGym",
     max_episode_steps=300,
)


# Yours

register(
     id="MyEnv",
     entry_point="gym_skeletons.environments.my_env:MyEnvGym",
     max_episode_steps=300,
)




