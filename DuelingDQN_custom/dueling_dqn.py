import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np

from network import DuelingDeepQNetwork
from buffer import ReplayBuffer

class Agent():
    def __init__(self, input_dims, env, epsilon=1, lr=1e-3, gamma=0.99, n_actions=2, batch_size=64,
                 epsilon_dec=1e-3, eps_end=0.01, 
                 mem_size=100000, fc1_dims=128,
                 fc2_dims=128, replace=100):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 1
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        self.action_space = [i for i in range(n_actions)]

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation, evaluate=False):
        if np.random.random() < self.epsilon and evaluate is False:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states) 
        q_next = self.q_next(states_) 
        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        
        for idx, terminal in enumerate(dones):
            q_target[idx, actions[idx]] = rewards[idx] * 10 + \
                    self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min

        self.learn_step_counter += 1

    def save_models(self, i):
        print('... saving models ...')
        self.q_eval.save_weights("./models/" + str(i) + "_eval.h5")
        self.q_next.save_weights("./models/" + str(i) + "_target.h5")

    def load_models(self, i):
        print('... loading models ...')
        self.q_eval.load_weights("./models/" + str(i) + "_eval.h5")
        self.q_next.load_weights("./models/" + str(i) + "_target.h5")