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


n_experiments = 1
T = 1000
environments = []

environments.append(chain.Chain(5))


algs = []
algs.append(MDPBelief.SimpleModelBasedRL)
algs.append(MDPBelief.SampleBasedRL)
n_algs = len(algs)
reward_t = np.zeros([T, n_algs])
total_reward = np.zeros([n_algs])
for experiment in range(n_experiments):
  env = environments[experiment]
  env.reset()
  alg_index = 0
  for Alg in algs:
    alg = Alg(env.observation_space.n, env.action_space.n)
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
print(total_reward)
plt.clf()
plt.plot(moving_average(reward_t, 10))
plt.legend(["QLearning"])
plt.savefig("qlearning.pdf")
#  plt.show()
  
 


