import tensorflow as tf

import numpy as np
from models import CategoricalActor, VNN
from gymnasium.spaces import Box, Discrete

import os

# disable gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class A2C(tf.keras.Model):
    """
    Unscaled version (runs on 1 cpu).
    """

    def __init__(self, observation_space, action_space, lr=1e-4, discount=0.99):

        if isinstance(observation_space, Box):
            state_dim = observation_space.shape[0]
        else: # we are most likely discrete, might need to be converted to one_hots
            state_dim = 1
        num_actions = action_space.n
        super(A2C, self).__init__(name='A2C')
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.discount = discount

        # TODO:
        # What is the critic ? what is the actor ?
        # How to build batches, how to do gradient descent with tensorflow ?

    def act(self, state):
        # Use the actor
        pass

    def update(self, state, action, reward, new_state, done, log_name=""):
        # Build batch, then push it to the learning function
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

        return {}

    def reset(self, state):
        # Nothing to do, we reset by calling DQL(args)
        pass


class SimpleBuffer:
    """
    Fill buffer, use data, restart
    """

    def __init__(self, observation_dim, batch_size=64):
        self.idx = 0
        self.size = batch_size
        self.observation_dim = observation_dim

        # TODO

    def record(self, state, action, reward, new_state, done):
        pass

    def get_batch(self):
        pass
