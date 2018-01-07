import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np


class DDPG():
    def __init__(self, name, layer_norm=True, nb_actions=2, nb_input=16):
        gamma = 0.999
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        state_input = tf.placeholder(shape=[None, nb_input], dtype=tf.float32)
        action_input = tf.placeholder(shape=[None, nb_actions], dtype=tf.float32)
        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        def actor_network(name):
            with tf.variable_scope(name) as scope:
                x = state_input
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, self.nb_actions,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.tanh(x)
            return x

        def critic_network(name, action_input, reuse=False):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()

                x = state_input
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.concat([x, action_input], axis=-1)
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            return x

        self.action_output = actor_network(name + '_actor')
        self.critic_output = critic_network(name + '_critic', action_input=action_input)
        self.state_input = state_input
        self.action_input = action_input
        self.reward = reward

        self.actor_optimizer = tf.train.AdamOptimizer(1e-4)
        self.critic_optimizer = tf.train.AdamOptimizer(1e-3)

        self.actor_loss = -tf.reduce_mean(critic_network(name + '_critic', action_input=self.action_output, reuse=True))
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss)

        #avs = self.actor_optimizer.compute_gradients(self.actor_loss)
        #aapped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in avs if grad is not None]
        #self.actor_train = self.actor_optimizer.apply_gradients(aapped_gvs)

        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic_output))
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss)

        #cvs = self.critic_optimizer.compute_gradients(self.critic_loss)
        #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in cvs if grad is not None]
        #self.critic_train = self.actor_optimizer.apply_gradients(capped_gvs)

    def train_actor(self, state, sess):
        sess.run(self.actor_train, {self.state_input: state})

    def train_critic(self, state, action, target, sess):
        sess.run(self.critic_train, {self.state_input: state, self.action_input: action, self.target_Q: target})

    def action(self, state, sess):
        return sess.run(self.action_output, {self.state_input: state})

    def Q(self, state, action, sess):
        return sess.run(self.critic_output, {self.state_input: state, self.action_input: action})