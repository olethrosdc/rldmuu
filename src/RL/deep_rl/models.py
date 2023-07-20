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

        self.dist = Categorical(dim=action_dim)
        self.action_dim = action_dim

        self.core = [Dense(64, activation="relu", dtype="float32", name='dense_%d' % i) for i in range(2)]

        self.prob = Dense(action_dim, dtype='float32', name="prob", activation="softmax")

        # We simply initialize the network parameters by calling the network.
        self.get_action(np.zeros((1, state_dim), dtype=np.float32))

    def _compute_feature(self, features):

        for layer in self.core:
            features = layer(features)

        return features

    def _compute_dist(self, states):
        """
        Compute categorical distribution

        :param states (np.ndarray or tf.Tensor): Inputs to neural network.
            NN outputs probabilities of K classes
        :return: Categorical distribution
        """

        features = self._compute_feature(states)

        probs = self.prob(features)
        return probs

    def get_action(self, state):
        single_state = len(state.shape) == 1
        if not isinstance(state, np.ndarray):
            state = np.array([[state]])
        else:
            missing_dims = 3 - len(state.shape)
            for missing_dim in range(missing_dims):
                state = state[np.newaxis]

        action, probs = self._get_action_body(tf.constant(state))


        return action.numpy()[0, 0] if single_state else action

    @tf.function
    def _get_action_body(self, state):
        probs = self._compute_dist(state)
        action = tf.squeeze(self.dist.sample(probs), axis=1)
        return action, probs

    def get_probs(self, states):
        return self._compute_dist(states)

    def compute_entropy(self, states):
        param = self._compute_dist(states)
        return self.dist.entropy(param)


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

        # Input
        self.core = [Dense(64, activation='relu', dtype='float32', name="core_%d" % i) for i in range(2)]
        # Output
        # Our output is unbounded, -inf to inf, thus the linear activation:
        self.v = Dense(1, activation='linear', dtype='float32', name="output")

        # The first dimension if for batches
        # We simply initialize the network parameters by calling the network.
        self.call(np.zeros((1, state_dim), dtype=np.float32))

    def call(self, states):
        """
        Pass forward through the layers.
        """

        features = states
        for layer in self.core:
            features = layer(features)

        return self.v(features)


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

        # Input
        self.core = [Dense(32, activation='relu', dtype='float32', name="core_%d" % i) for i in range(2)]
        # Output
        # Our output is unbounded, -inf to inf, thus the linear activation:
        self.q = Dense(num_actions, activation='linear', dtype='float32', name="output")

        # We simply initialize the network parameters by calling the network.
        self.call(np.zeros((1, state_dim), dtype=np.float32))

    def call(self, states):
        """
        Pass forward through the layers.
        """

        features = states
        for layer in self.core:
            features = layer(features)

        return self.q(features)


class QPolicy(QNN):
    def __init__(self, state_dim, num_actions, name='q_function'):
        super().__init__(state_dim, num_actions, name)

    def act(self, states):
        if not isinstance(states, np.ndarray):
            states = np.array([states])
        if len(states.shape) < 2:
            states = states[np.newaxis]
            return np.argmax(self(states))
        else:
            return np.argmax(self(states), axis=-1)