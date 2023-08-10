import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Activation, Concatenate, Input
from tensorflow.keras.optimizers.legacy import Adam

# from rl.agents import DDPGAgent
from rl.policy import Policy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.processors import MultiInputProcessor

from keras_ddpg import Agent

class ENV(Env):
    def __init__(self):
        self.action_space = Discrete(5, start=-2)
        # self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.observation_space = Box(low=0, high=100, dtype=np.int8)
        # self.state = np.array([38 + random.randint(-5,5)], dtype=object)
        self.state = 38 + random.randint(-5,5)
        self.shower_length = 60

    def step(self, action):
        """
        Training forward function
        - 실제 simulation 환경에서 fan 속도 제어 명령을 보낼 곳
        - reward를 받아올 때까지 시간 delay가 필요함

        Args:
            action (int, sample): 팬 속도 제어

        Returns:
            state: _description_
            reward: action에 따른 보상
            done: episode 종료 여부 표시
            info: .
        """
        self.state += action -1 
        self.shower_length -= 1 
        
        # Calculating the reward
        if self.state[len(self.state)-1] >=37 and self.state[len(self.state)-1] <=39: 
            reward =1 
        else: 
            reward = -1 
        
        # Checking if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Setting the placeholder for info
        info = {}
        # Returning the step information
        return self.state, reward, done, info
    
    def get_obs(self):
        observation = np.array([self.state], dtype=np.int8)
        print("         getobs:", observation, observation.shape)
        return observation
    
    def reset(self):
        # self.state = np.array([38 + random.randint(-5,5)], dtype=object)
        
        self.state = 38 + random.randint(-5,5)
        self.shower_length = 60 
        return self.state #self.get_obs()


def build_actor(env, nb_actions):
    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=env.observation_space.shape))
    actor.add(Dense(12))
    actor.add(Activation('relu'))
    actor.add(Dense(12))
    actor.add(Activation('relu'))
    actor.add(Dense(12))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())
    return actor

def build_critic(action_input, observation_input):
    x = Concatenate()([action_input, observation_input])
    x = Dense(24)(x)
    x = Activation('relu')(x)
    x = Dense(24)(x)
    x = Activation('relu')(x)
    x = Dense(24)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())
    return critic


env = ENV()
nb_actions = env.action_space.n

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)

actor = build_actor(env, nb_actions)
critic = build_critic(action_input, observation_input)

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)

# agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
#                   memory=memory, nb_steps_warmup_critic=10, nb_steps_warmup_actor=10,
#                   random_process=random_process, gamma=.99, target_model_update=1e-3, is_discrete=True) #, processor=processor)
# agent = ddpgAgent(env, is_discrete=True, batch_size=12, actor=actor, critic=critic)
agent = Agent(input_dims=[1, 1], env=env, n_actions=nb_actions, fc1=12, fc2=24)

# agent.compile(Adam(learning_rate=.001, clipnorm=1.), metrics=['mae'])
# agent.fit(env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=200)
