import tensorflow as tf

import numpy as np
from models import QPolicy
from gymnasium.spaces import Box, Discrete

import os

# disable gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class DQL(tf.keras.Model):
    def __init__(self, observation_space, action_space, lr=1e-3, discount=0.9, target_update_period=1,
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

        self.qnn = QPolicy(state_dim, num_actions)
        self.target_qnn = QPolicy(state_dim, num_actions)
        # Copy the weights
        self.target_qnn.set_weights(self.qnn.get_weights())

        self.optim = tf.keras.optimizers.Adam(lr)

        self.replay_buffer = ReplayBuffer(state_dim, int(1e5))

        self.train_step_idx = 0


    def act(self, state):
        # The behaviour of our currently learning policy
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return self.qnn.act(state)

    def update(self, state, action, reward, new_state, done, log_name=""):
        # Save data in the replay buffer, sample a batch and learn from it

        self.replay_buffer.record(state, action, reward, new_state, done)
        batch = self.replay_buffer.sample(sample_size=4)
        if batch is not None:
            states, actions, rewards, new_states, dones = batch

            loss = self._train(states, actions, rewards, new_states, dones, gpu=-1)

            tf.summary.scalar(name=log_name + "/loss", data=loss)

        # epsilon decay
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
        tf.summary.scalar(name=log_name + "/epsilon", data=self.epsilon)

        self.train_step_idx += 1
        # Update the target network every target_update_period training steps
        if self.train_step_idx % self.target_update_period == 0:
            self.target_qnn.set_weights(self.qnn.get_weights())

    @tf.function
    def _train(self, states, actions, rewards, new_states, dones, gpu):
        '''
        inner training function, arguments must be in datatypes recognized by tensorflow.

        tf.function decorator in Tensorflow2 speeds up execution. Compiles the code with C++
        Main training function
        Deals with array differently that python, make sure to use tensorflow methods to handle arrays.
        '''

        batch_size = len(states)

        # Possible to execute on a gpu, but for simplicity we do it on cpu with gpu =  -1
        with tf.device("/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"):

            # Here the tape will track everything related to gradient descent
            with tf.GradientTape() as tape:

                # Should be of shape (batch_size, action_dim)
                current_q = self.qnn(states)  # Q[s]

                indexed_actions = tf.concat(
                    [
                        tf.expand_dims(tf.range(0, batch_size), axis=1), actions
                    ], axis=1
                )
                # indexed_actions is of shape (batch_size, 2)
                # indexed_actions[i] -> index in the batch, index of the action

                current_q_sa = tf.gather_nd(current_q, indexed_actions) # Q[s,a]
                # shape is (batch_size, 1)

                next_q = self.target_qnn(new_states)

                # axis -1 is the same as taking the last axis (here its 1)
                # we take the max for each sample
                next_q_max = tf.reduce_max(next_q, axis=-1)
                # next_q_max is of shape (batch_size, 1)

                # Q = Q + alpha * (next_Q_max * discount + reward - Q)

                td_error = (1. - dones) * (self.discount * next_q_max) + rewards - current_q_sa

                loss = tf.reduce_mean(tf.square(td_error))

            gradient = tape.gradient(loss, self.qnn.trainable_variables)

            # Update the weights
            self.optim.apply_gradients(zip(gradient, self.qnn.trainable_variables))

        return loss

    def reset(self, state):
        # Nothing to do, we reset by calling DQL(args)
        pass


class ReplayBuffer:

    def __init__(self, state_dim, size=5000):
        self.idx = 0
        self.state_dim = state_dim
        self.size = size

        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.new_states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, 1), dtype=np.int32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)


    def record(self, state, action, reward, new_state, done):
        # save the data
        current_idx = self.idx % self.size

        self.states[current_idx] = state
        self.new_states[current_idx] = new_state
        self.actions[current_idx] = action
        self.rewards[current_idx] = reward
        self.dones[current_idx] = done

        self.idx += 1

    def sample(self, sample_size=256):
        # Wait until enough data in the buffer, then sample uniformly from it
        if self.idx  >= sample_size * 10:

            # You dont want to sample indexes where self.data[idx] is not set yet
            max_idx = min(self.idx-1, self.size-1)

            sampled_idxs = np.random.choice(max_idx, sample_size, replace=True)

            # a = [0,1,4]
            # b = np.array([1,2,3,4,5])
            # b[a] -> np.array([1,2,5])

            # DATA is of shape (sample_size, data_size)
            return (
                self.states[sampled_idxs],
                self.actions[sampled_idxs],
                self.rewards[sampled_idxs],
                self.new_states[sampled_idxs],
                self.dones[sampled_idxs]
            )
        else:
            return None

