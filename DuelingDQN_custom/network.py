import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions=5, fc1_dims=512, fc2_dims=512):
        super(DuelingDeepQNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join(
                    '_ddpg')

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_action = n_actions

        self.fc1 = Dense(self.fc1_dims, input_dim=1, activation='relu')
        self.fc2 = Dense(self.fc2_dims, input_dim=fc1_dims, activation='relu')
        self.V = Dense(1, activation=None)
        self.A = Dense(n_actions, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    def advantage(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        A = self.A(x)

        return A