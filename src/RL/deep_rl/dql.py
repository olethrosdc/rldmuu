import tensorflow as tf

import numpy as np
from models import QPolicy
from gymnasium.spaces import Box, Discrete

import os

# disable gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



class DQL(tf.keras.Model):
    def __init__(self, observation_space, action_space, lr=1e-4, discount=0.99, target_update_period=128,
                 epsilon_decay=0.999, epsilon_min=0.01):

        if isinstance(observation_space, Box):
            state_dim = observation_space.shape[0]
        else: # we are discrete, might need to be converted to one_hots
            state_dim = 1
        num_actions = action_space.n
        super(DQL, self).__init__(name='DQL')
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.eps_decay = epsilon_decay
        self.eps_min = epsilon_min
        self.epsilon = 1.

        # TODO
        # Instantiate the DQN, optimizer, etc.
        # We need to use a replay buffer and a target network for stability.


    def act(self, state):
        # The behaviour of our currently learning policy
        pass

    def update(self, state, action, reward, new_state, done, log_name=""):
        # Save data in the replay buffer, sample a batch and learn from it
        pass


    @tf.function
    def _train(self, state, action, reward, new_state, done, gpu):
        '''
        inner training function, arguments must be in datatypes recognized by tensorflow.

        tf.function decorator in Tensorflow2 speeds up execution. Compiles the code with C++
        Main training function
        Deals with array differently that python, make sure to use tensorflow methods to handle arrays.
        '''

        # Possible to execute on a gpu, but for simplicity we do it on cpu with gpu =  -1
        with tf.device("/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"):
            pass

        return 0

    def reset(self, state):
        # Nothing to do, we reset by calling DQL(args)
        pass


class ReplayBuffer:

    def __init__(self, observation_dim, size=5000):
        self.idx = 0
        self.observation_dim = observation_dim
        self.size = size

    def record(self, state, action, reward, new_state, done):
        # save the data
        pass

    def sample(self, sample_size=256):
        # Wait until enough data in the buffer, then sample uniformly from it
        pass

