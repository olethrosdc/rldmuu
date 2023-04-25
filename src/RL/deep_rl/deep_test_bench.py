
import numpy as np
import matplotlib.pyplot as plt
import dql
import actor_critic
from copy import deepcopy
import datetime

import tensorflow as tf

# Tensorboard Logging
tf.summary.experimental.set_step(0)
log_dir = 'logs/' + str(datetime.datetime.now())
writer = tf.summary.create_file_writer(log_dir)
writer.set_as_default()

def moving_average(x, K):
  T = x.shape[0]
  n = x.shape[1]
  m = int(np.ceil(T / K))
  y = np.zeros([m, n])
  for alg in range(n):
      for t in range(m):
        y[t,alg] = np.mean(x[t*K:(t+1)*K, alg])
  return y


n_experiments = 1
T = 1000
n_episodes = 10000
environments = []

import gymnasium as gym
#environments.append(gym.make("FrozenLake-v1", is_slippery=False))#, render_mode="human"))
environments.append(gym.make("CartPole-v1"))
#environments.append(gym.make("LunarLander-v2"))
#environments.append(chain.Chain(4))



algs = []
hyper_params = []
#algs.append(QLearning.QLearning)
#algs.append(dql.DQL)
algs.append(actor_critic.A2C)


#algs.append(ModelBasedRL.GreedyQiteration)
#algs.append(ModelBasedRL.ModelBasedQLearning)
#algs.append(MDPBelief.ExpectedMDPHeuristic)
#algs.append(MDPBelief.SampleBasedRL)
n_algs = len(algs)


reward_i = np.zeros([n_episodes, n_algs])
reward_t = np.zeros([T, n_algs])
total_reward = np.zeros([n_algs])

for experiment in range(n_experiments):
  print(experiment)
  env = environments[0]
  alg_index = 0
  for Alg in algs:

    alg = Alg(env.observation_space, env.action_space)

    tf.summary.experimental.set_step(0)
    i = 0
    for episode in range(n_episodes):

      run_reward = 0
      observation, info = env.reset()

      alg.reset(observation)
      done = False
      t = 0
      while not done and t < T:
        action = alg.act(observation)
        env.render()

        new_observation, reward, truncated, done, info = env.step(action)
        done = done or truncated

        alg.update(deepcopy(observation), action, reward, deepcopy(new_observation), done, log_name=f'{alg.__class__}')

        observation = deepcopy(new_observation)

        run_reward += reward
        reward_t[t, alg_index] += reward
        t += 1
        i += 1
        tf.summary.experimental.set_step(i)
      reward_i[episode, alg_index] += run_reward

      mean_score = np.mean(reward_i[episode-25:episode, alg_index])
      print("Episode", episode, "Average reward:", mean_score )

      tf.summary.scalar(name=f"{alg.__class__}/average_episode_reward", data=mean_score)

      total_reward[alg_index] += run_reward

  alg_index += 1
  env.close()

total_reward /= n_experiments
reward_t /= n_experiments
reward_i /= n_experiments

print(total_reward)

plt.plot(reward_i, label=[alg.__name__ for alg in algs])

plt.legend()
plt.show()
  
  
  
 


