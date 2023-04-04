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
        

