# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow
Tested in Tensorflow 1.4 and 1.5

@author: Xiang Zhong
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


class PolicyValueNet():
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height

        self.is_training = tf.placeholder(tf.bool)
        # Define the tensorflow neural network
        # 1. Input:
        self.input_states = tf.placeholder(
                tf.float32, shape=[None, 16, board_height, board_width])
        self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1])
        
        # 2. Common Networks Layers
        self.block1 = self._block(self.input_state, 32, 3, is_training=self.is_training, scope="block1")
        self.block2 = self._block(self.block1, 64, 3, is_training=self.is_training, scope="block2")
        self.block3 = self._block(self.block2, 128, 3, is_training=self.is_training, scope="block3")
        
        # 3-1 Action Networks
        self.action_conv = tf.layers.conv2d(inputs=self.block3, filters=4,
                                            kernel_size=[1, 1], padding="same",
                                            data_format="channels_last",
                                            activation=tf.nn.relu)
        # Flatten the tensor
        self.action_conv_flat = tf.reshape(
                self.action_conv, [-1, 4 * board_height * board_width])
        # 3-2 Full connected layer, the output is the log probability of moves
        # on each slot on the board
        self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                         units=board_height * board_width,
                                         activation=tf.nn.log_softmax)
        # 4 Evaluation Networks
        self.evaluation_conv = tf.layers.conv2d(inputs=self.block3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                data_format="channels_last",
                                                activation=tf.nn.relu)
        self.evaluation_conv_flat = tf.reshape(
                self.evaluation_conv, [-1, 2 * board_height * board_width])
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu)
        # output the score of evaluation on current state
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1, activation=tf.nn.tanh)

        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_fc2
        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                       self.evaluation_fc2)
        # 3-2. Policy Loss function
        self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, board_height * board_width])
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)

    def _batch_norm(self, x, is_training, scope="bn"):
        z = tf.cond(is_training, lambda: batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,is_training=True, reuse=None, trainable=True, scope=scope), 
                                lambda: batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,is_training=False, reuse=True, trainable=False, scope=scope))
        return z

    def _block(self, x, n_out, n, is_training, scope="block"):
        with tf.variable_scope(scope):
            out = self._bottleneck(x, n_out, is_training, scope="bottleneck1")
            for i in range(1, n):
                out = self._bottleneck(out, n_out, is_training, scope=("bottleneck%s" % (i + 1)))
            return out

    def _bottleneck(self, x, n_out, is_training, scope="bottleneck"):
        """ A residual bottleneck unit"""
        n_in = x.get_shape()[-1]

        with tf.variable_scope(scope):
            h = tf.layers.conv2d(inputs=x, filters=n_out, kernel_size=[3, 3], padding="same", data_format="channels_last", activation=None)
            h = self._batch_norm(h, is_training, scope="bn_1")
            h = tf.nn.relu(h)
            h = tf.layers.conv2d(inputs=h, filters=n_out, kernel_size=[3, 3], padding="same", data_format="channels_last", activation=None)
            h = self._batch_norm(h, is_training, scope="bn_2")
            h = tf.nn.relu(h)
            h = tf.layers.conv2d(inputs=h, filters=n_out, kernel_size=[3, 3], padding="same", data_format="channels_last", activation=None)
            h = self._batch_norm(h, is_training, scope="bn_3")

            if n_in != n_out:
                shortcut = tf.layers.conv2d(inputs=x, filters=n_out, kernel_size=[1, 1], padding="same", data_format="channels_last", activation=None)
                shortcut = self._batch_norm(shortcut, is_training, scope="bn_4")
            else:
                shortcut = self._batch_norm(x, is_training, scope="bn_4")
            return tf.nn.relu(self._batch_norm(shortcut + h, is_training, scope="bn_5"))
            
    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probs, value = self.session.run(
                [self.action_fc, self.evaluation_fc2],
                feed_dict={self.input_states: state_batch,
                            self.is_training: False}
                )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 16, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr,
                           self.is_training: True})
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
