import gym
import gym_bandits
env = gym.make("BanditTenArmedGaussian-v0")
env.reset()
for i_episode in range(100):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

