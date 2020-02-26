import numpy as np
import tensorflow as tf
from tensorflow import keras
from .base_policy import BasePolicy
from cs285.infrastructure.tf_utils import build_mlp
import tensorflow_probability as tfp

class MLPPolicy(BasePolicy):

    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        learning_rate=1e-4,
        training=True,
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        # self.sess = sess
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.model = build_mlp(output_size=self.ac_dim, n_layers=self.n_layers, size=self.size)
        self.loss_object = lambda y_true, y_pred:tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        '''
        # saver for policy variables that are not related to training
        self.policy_vars = [v for v in tf.all_variables() if policy_scope in v.name and 'train' not in v.name]
        self.policy_saver = tf.train.Saver(self.policy_vars, max_to_keep=None)
        '''
    ##################################

    def action_sample(self, obs):
        mean = self.model(obs)
        logstd = tf.Variable(tf.zeros(self.ac_dim), name='logstd')
        self.sample_ac = mean + tf.exp(logstd) * tf.random_normal(tf.shape(mean), 0, 1)

    ##################################

    '''
    def save(self, filepath):
        self.policy_saver.save(self.sess, filepath, write_meta_graph=False)

    def restore(self, filepath):
        self.policy_saver.restore(self.sess, filepath)
    '''

    ##################################

    # update/train this policy
    def update(self, observations, actions):
        raise NotImplementedError

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):

    """
        This class is a special case of MLPPolicy,
        which is trained using supervised learning.
        The relevant functions to define are included below.
    """

    def update(self, observations, actions):
        assert(self.training, 'Policy must be created with training=True in order to perform training updates...')
        
        with tf.GradientTape() as tape:
            loss_value = self.loss_object(self.model(observations), actions)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

