import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam

Adam._name = 'AdamOptimizer'

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from custom_env import ENV
from networks import build_agent, build_model

if __name__ == "__main__":

    evaluate = bool(input("[is evaluation? True / False]: "))
    env = ENV()

    states = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(states, actions)
    print(model.summary())

    dqn = build_agent(model, actions)
    if evaluate is True:
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        dqn.load_weights('./dqn/20_epoch')
    # dqn.fit(env, nb_steps=200000, visualize=False, verbose=1)

        results = dqn.test(env, nb_episodes=1, visualize=False)
        print(np.mean(results.history['episode_reward']))

    else:
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        dqn.fit(env, nb_steps=200000, visualize=False, verbose=1)

        results = dqn.test(env, nb_episodes=150, visualize=False)
        print(np.mean(results.history['episode_reward']))
        dqn.save_weights('./dqn/20_epoch')