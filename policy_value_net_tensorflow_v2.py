# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow V2
"""


from stat import FILE_ATTRIBUTE_ENCRYPTED
from tabnanny import verbose
import numpy as np
import tensorflow as tf

print( tf.__version__ )




def create_model(board_width, board_height):

    class RenjuModel(tf.Module):
        def __init__(self):
            l2_penalty_beta = 1e-4

            # Define the tensorflow neural network
            # 1. Input:
            self.inputs = tf.keras.Input( shape=(4, board_height, board_width), dtype=tf.dtypes.float32)
            self.transposed_inputs = tf.keras.layers.Lambda( lambda x: tf.transpose(x, [0, 2, 3, 1]) )(self.inputs)

            # 2. Common Networks Layers
            self.conv1 = tf.keras.layers.Conv2D( name="conv1",
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                data_format="channels_last",
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.transposed_inputs)

            self.conv2 = tf.keras.layers.Conv2D( name="conv2", 
                filters=64, 
                kernel_size=(3, 3), 
                padding="same", 
                data_format="channels_last", 
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.conv1)

            self.conv3 = tf.keras.layers.Conv2D( name="conv3",
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                data_format="channels_last",
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.conv2)

            # 3-1 Action Networks
            self.action_conv = tf.keras.layers.Conv2D( name="action_conv",
                filters=4,
                kernel_size=(1, 1),
                padding="same",
                data_format="channels_last",
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.conv3)

            # flatten tensor
            self.action_conv_flat = tf.keras.layers.Reshape( (-1, 4 * board_height * board_width), name="action_conv_flat" 
            )(self.action_conv)

            # 3-2 Full connected layer, the output is the log probability of moves
            # on each slot on the board
            self.action_fc = tf.keras.layers.Dense( board_height * board_width,
                activation=tf.nn.log_softmax,
                name="action_fc",
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.action_conv_flat)

            # 4 Evaluation Networks
            self.evaluation_conv = tf.keras.layers.Conv2D( name="evaluation_conv",
                filters=2,
                kernel_size=(1, 1),
                padding="same",
                data_format="channels_last",
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta))(self.conv3)

            self.evaluation_conv_flat = tf.keras.layers.Reshape( (-1, 2 * board_height * board_width),
                name="evaluation_conv_flat" 
                )(self.evaluation_conv)

            self.evaluation_fc1 = tf.keras.layers.Dense( 64,
                activation=tf.keras.activations.relu,
                name="evaluation_fc1",
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.evaluation_conv_flat)

            self.evaluation_fc2 = tf.keras.layers.Dense( 1, 
                activation=tf.keras.activations.tanh,
                name="evaluation_fc2",
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.evaluation_fc1)

            self.model = tf.keras.Model(inputs=self.inputs, outputs=[self.action_fc, self.evaluation_fc2], name="renju_model")
            self.model.summary()

            self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=[self.action_loss, tf.keras.losses.MeanSquaredError()],
                    metrics=['accuracy'])


        @tf.function(input_signature=[ tf.TensorSpec([None, board_height * board_width], tf.float32),
            tf.TensorSpec([None, None, board_height * board_width], tf.float32)
        ])
        def action_loss(self, labels, predictions):
            # labels are probabilities; predictions are logits
            return tf.negative(tf.reduce_mean(
                        tf.reduce_sum(tf.multiply(labels, predictions[0]), 1)))
           

        @tf.function
        def train(self, x, y):
            with tf.GradientTape() as tape:
                predictions = self.model(x, training=True)  # Forward pass
                # the loss function is configured in `compile()`
                loss = self.model.compiled_loss(y, predictions, regularization_losses=self.model.losses)
 
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

            entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(predictions[0][0]) * predictions[0][0], 1)))
            result = {"loss": loss, "entropy" : entropy}
            return result

        @tf.function(input_signature=[
            tf.TensorSpec([None, 4, board_height, board_width], tf.float32),
        ])
        def predict(self, x):
            return self.model(x)

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def save(self, checkpoint_path):
            tensor_names = [weight.name for weight in self.model.weights]
            tensors_to_save = [weight.read_value() for weight in self.model.weights]
            tf.raw_ops.Save(
                filename=checkpoint_path, tensor_names=tensor_names,
                data=tensors_to_save, name='save')
            return {
                "checkpoint_path": checkpoint_path
            }

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def restore(self, checkpoint_path):
            restored_tensors = {}
            for var in self.model.weights:
                restored = tf.raw_ops.Restore( file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype, name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
            return restored_tensors

    return RenjuModel()


class PolicyValueNet():
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height

        
        self.model = create_model(board_width=board_width, board_height=board_height)

        if model_file is not None:
            self.restore_model(model_file)


    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        [log_act_probs, value] = self.model.predict(state_batch)
        act_probs = np.exp(log_act_probs)
        return act_probs[0], value[0].numpy()


    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = tf.reshape(winner_batch, (-1, 1))

        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        mcts_probs = tf.convert_to_tensor(mcts_probs, dtype=tf.float32)
        winner_batch = tf.cast(winner_batch, dtype=tf.float32)
        history = self.model.train( state_batch, [mcts_probs, winner_batch] )
        loss = history['loss'].numpy()
        entropy = history['entropy'].numpy()

        return loss, entropy

    def save_model(self, model_path):
        self.model.save(model_path)

    def restore_model(self, model_path):
        self.model.restore(model_path)
