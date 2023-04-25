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
        self.critic = VNN(state_dim)
        self.actor = CategoricalActor(state_dim, num_actions)
        
        self.optim = tf.keras.optimizers.Adam(lr)

        self.data_buffer = SimpleBuffer(state_dim, 128)

    def act(self, state):
        return self.actor.get_action(state)

    def update(self, state, action, reward, new_state, done, log_name=""):
        # Build batch, then push it to the learning function

        self.data_buffer.record(state, action, reward, new_state, done)

        batch = self.data_buffer.build_batch()

        if batch is not None:
            result_dict = self._train(*batch, gpu=-1)

            for key, value in result_dict.items():
                tf.summary.scalar(name=log_name + "/" + key, data=value)



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
            with tf.GradientTape() as tape:

                # We need the current V, next V, the logprobs of the taken actions

                current_v = self.critic(states)
                next_v = self.critic(new_states)
                advantage = (1.-dones) * tf.stop_gradient(rewards + next_v * self.discount) - current_v

                # value loss -> (A)^2
                value_loss = tf.reduce_mean(tf.square(advantage))

                probs = self.actor.get_probs(states)

                indexed_actions = tf.concat(
                    [
                        tf.expand_dims(tf.range(0, batch_size), axis=1), actions
                    ], axis=1
                )

                pi_sa = tf.gather_nd(probs, indexed_actions)  # Pi[s,a]

                log_pi_sa = tf.math.log(pi_sa + 1e-8)

                policy_loss = - tf.reduce_mean(log_pi_sa * tf.stop_gradient(advantage))

                loss = policy_loss + 0.5 * value_loss

            gradient = tape.gradient(loss, self.critic.trainable_variables + self.actor.trainable_variables)
            # Update the weights
            self.optim.apply_gradients(zip(gradient, self.critic.trainable_variables + self.actor.trainable_variables))

            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": tf.reduce_mean(entropy)
        }

    def reset(self, state):
        # Nothing to do, we reset by calling DQL(args)
        pass


class SimpleBuffer:
    """
    Fill buffer, use data, restart
    """

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

    def build_batch(self):
        # Wait until enough data in the buffer, then sample uniformly from it
        if self.idx == self.size:

            # On "vide" le buffer
            self.idx = 0

            return (
                self.states,
                self.actions,
                self.rewards,
                self.new_states,
                self.dones
            )
        else:
            # We wait for the buffer to be filled
            return None

