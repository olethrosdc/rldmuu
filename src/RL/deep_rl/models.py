import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.backend import set_value
import numpy as np
from tensorflow.keras.activations import relu, softmax
import itertools
from copy import deepcopy


class Distribution:
    def __init__(self, dim):
        self._dim = dim
        self._tiny = 1e-8

    @property
    def dim(self):
        raise self._dim

    def kl(self, old_dist, new_dist):
        """
        Compute the KL divergence of two distributions
        """
        raise NotImplementedError

    def likelihood_ratio(self, x, old_dist, new_dist):
        raise NotImplementedError

    def entropy(self, dist):
        raise NotImplementedError

    def log_likelihood_sym(self, x, dist):
        raise NotImplementedError

    def log_likelihood(self, xs, dist):
        raise NotImplementedError


class Categorical(Distribution):
    def kl(self, old_prob, new_prob):
        """
        Compute the KL divergence of two Categorical distribution as:
            p_1 * (\log p_1  - \log p_2)
        """
        return tf.reduce_sum(
            old_prob * (tf.math.log(old_prob + self._tiny) - tf.math.log(new_prob + self._tiny)))

    def likelihood_ratio(self, x, old_prob, new_prob):
        return (tf.reduce_sum(new_prob * x) + self._tiny) / (tf.reduce_sum(old_prob * x) + self._tiny)

    def log_likelihood(self, x, param):
        """
        Compute log likelihood as:
            \log \sum(p_i * x_i)

        :param x (tf.Tensor or np.ndarray): Values to compute log likelihood
        :param param (Dict): Dictionary that contains probabilities of outputs
        :return (tf.Tensor): Log probabilities
        """
        probs = param["prob"]
        assert probs.shape == x.shape, \
            "Different shape inputted. You might have forgotten to convert `x` to one-hot vector."
        return tf.math.log(tf.reduce_sum(probs * x, axis=1) + self._tiny)

    def sample(self, probs, amount=1):
        # NOTE: input to `tf.random.categorical` is log probabilities
        # For more details, see https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/random/categorical
        # [probs.shape[0], 1]
        # tf.print(probs, tf.math.log(probs), tf.random.categorical(tf.math.log(probs), amount), summarize=-1)
        return tf.cast(tf.map_fn(lambda p: tf.cast(tf.random.categorical(tf.math.log(p), amount), tf.float32), probs),
                       tf.int64)

    def entropy(self, probs):
        return -tf.reduce_sum(probs * tf.math.log(probs + self._tiny), axis=1)


class CategoricalActor(tf.keras.Model):
    '''
    Policy network model class
    '''

    def __init__(self, state_dim, action_dim, name="CategoricalActor"):
        super().__init__(name=name)

        self.action_dim = action_dim

        # TODO
        # Build the network

    def _compute_feature(self, features):
        # Pass forward through the layers
        pass

    def _compute_dist(self, states):
        """
        Compute categorical distribution

        :param states (np.ndarray or tf.Tensor): Inputs to neural network.
            NN outputs probabilities of K classes
        :return: Categorical distribution
        """
        pass

    def get_action(self, state):
        # We have to handle both single states and batches

        pass

    @tf.function
    def _get_action_body(self, state):
        # inner function that does the computing
        # returns (action, probabilities)
        pass

    def get_probs(self, states):
        pass


class VNN(tf.keras.Model):
    """
    Value function model class
    Outputs one value
    """

    def __init__(self, state_dim, name='value_function'):
        """
        Typically, from features, 2 dense layers is enough
        Relu is the usual activation function for the core layers
        Fast computationally-wise, not derivable in 0.
        Relu function is as follows:
            outputs x if x > 0
            outputs 0 otherwise
        """

        super().__init__(name=name)

        # TODO
        # Build the network

    def call(self, states):
        """
        Pass forward through the layers.
        """
        pass


class QNN(tf.keras.Model):
    """
    Q function model class
    Outputs a value for each action in the given state.
    """

    def __init__(self, state_dim, num_actions, name='q_function'):
        """
        Same as V, but we output a value for each action
        """

        super().__init__(name=name)
        self.num_actions = num_actions

        # TODO
        # Build the network

    def call(self, states):
        """
        Pass forward through the layers.
        """
        pass


class QPolicy(QNN):
    def __init__(self, state_dim, num_actions, name='q_function'):
        super().__init__(state_dim, num_actions, name)

    def act(self, states):
        # If discrete observation
        if not isinstance(states, np.ndarray):
            states = np.array([states])

        # Check if we have the state in a batch or not
        # then return the action