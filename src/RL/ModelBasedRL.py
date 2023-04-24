import numpy as np
from MDP import DiscreteMDP
from ValueIteration import value_iteration

class ModelBasedRL:
    def __init__(self, n_states, n_actions, discount=0.9):
        self.n_actions = n_actions
        self.n_states = n_states
        self.discount = discount
        # boiler plate
    def act(self):
        return np.random.randint(self.n_actions)
    
    def update(self, action, reward, state):
        pass

    def reset(self, state):
        pass

    
class GreedyQIteration(ModelBasedRL):
    def __init__(self, n_states, n_actions, discount=0.9, epsilon=0.01, alpha = 0, decay = 0):
        super().__init__(n_states, n_actions, discount)
        self.epsilon = epsilon
        self.Q = np.zeros([n_states, n_actions])
        self.V = np.zeros(n_states)
        self.state = -1
        self.reset(self.state)
        # boiler plate
    def act(self):
        assert(self.state >= 0)
        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[self.state, :])

    # A simple count
    def update_model(self, action, reward, state):
        self.N_sas[self.state, action, state] +=1
        self.N_sa[self.state, action] +=1
        self.P_t[self.state,action,:] = self.N_sas[self.state, action, :] / self.N_sa[self.state, action]
        self.rho_t[self.state, action] += (reward - self.rho_t[self.state, action])/self.N_sa[self.state, action]
        #print("P_t\n")
        #print(100*self.P_t)
        
    ## change this to do Q-learning
    def update_q_function(self, action, reward, state):
        ## for the current state, and a random subset of states and
        ## actions, do a Q-learning-update.
        alpha = 0.1
        self.Q[self.state, action] += alpha * (reward + self.discount * max(self.Q[state, :]) - self.Q[self.state, action]) # simple Q-learning
        ## uniform sampling
        for k in range (100):
            s = int(np.random.choice(self.n_states))
            a = int(np.random.choice(self.n_actions))
            p = self.P_t[s,a,:]
            s_next = int(np.random.choice(self.n_states, p= p))
            self.Q[s, a] += alpha * (self.rho_t[s,a] + self.discount * max(self.Q[s_next, :]) - self.Q[s,a])
        ## proportional sampling
        P_sa = N_sa[s,a] / sum(sum(N_sa))
        # sampling from this
        for k in range (100):
            s, a = np.random.choice(P_sa) # not actually working code
            p = self.P_t[s,a,:]
            s_next = int(np.random.choice(self.n_states, p= p))
            self.Q[s, a] += alpha * (self.rho_t[s,a] + self.discount * max(self.Q[s_next, :]) - self.Q[s,a])

            
    def update(self, action, reward, state):
        self.update_model(action, reward, state)
        self.update_q_function(action, reward, state)
        self.state = state
        pass

    def reset(self, state):
        self.rho_t = np.ones([self.n_states, self.n_actions])
        self.N_sa = np.ones([self.n_states, self.n_actions]) * self.n_states
        self.N_sas = np.ones([self.n_states, self.n_actions, self.n_states])
        self.P_t = np.ones([self.n_states, self.n_actions, self.n_states]) / self.n_states
        self.state = state
        pass


