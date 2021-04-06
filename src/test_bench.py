import QLearning
import chain
import numpy as np
import matplotlib.pyplot as plt
import MDPBelief
def moving_average(x, K):
  T = x.shape[0]
  n = x.shape[1]
  m = int(np.ceil(T / K))
  y = np.zeros([m, n])
  for alg in range(n):
      for t in range(m):
        y[t,alg] = np.mean(x[t*K:(t+1)*K, alg])
  return y


n_experiments = 100
T = 1000
environments = []

environments.append(chain.Chain(5))


algs = []
algs.append(QLearning.QLearning)
n_algs = len(algs)

alpha = 0.4
epsilon = 0.3
decay = 0.1
for decay in np.arange(0,1,0.1):
  reward_t = np.zeros([T, n_algs])
  total_reward = np.zeros([n_algs])
  for experiment in range(n_experiments):
    env = environments[0];#experiment]
    env.reset()
    alg_index = 0
    for Alg in algs:
      alg = Alg(n_states = env.observation_space.n, n_actions = env.action_space.n, alpha = alpha, epsilon = epsilon, decay = decay)
      run_reward = 0
      for i_episode in range(1):
        observation = env.reset()
        alg.reset(observation)
        for t in range(T):
          env.render()
          action = alg.act()
          #print(observation, action)
          observation, reward, done, info = env.step(action)
          alg.update(action, reward, observation)
          run_reward += reward
          reward_t[i_episode, alg_index] += reward
          if done:
            #            print("Episode finished after {} timesteps".format(t+1))
            break
      total_reward[alg_index] += run_reward
      alg_index += 1
      env.close()
  total_reward /= n_experiments
  reward_t /= n_experiments
  print(alpha, epsilon, decay, total_reward)

  
 


