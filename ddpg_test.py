import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Activation, Concatenate, Input
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents import DDPGAgent
from rl.policy import Policy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.processors import MultiInputProcessor

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
        print("state 1: ", self.state)
        print("action : ", action)
        
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
        return self.get_obs(), reward, done, info
    
    def get_obs(self):
        observation = np.array([self.state], dtype=np.int8)
        print("         getobs:", observation, observation.shape)
        return observation
    
    def reset(self):
        # self.state = np.array([38 + random.randint(-5,5)], dtype=object)
        
        self.state = 38 + random.randint(-5,5)
        self.shower_length = 60 
        return self.get_obs()


def build_actor(env, nb_actions):
    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
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

def build_critic(action_input, flattened_observation, observation_input):
    x = Concatenate()([action_input, flattened_observation])
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

Adam._name = 'AdamOptimizer'
nb_actions = env.action_space.n
print(env.observation_space.shape)
print((1,) + env.observation_space.shape)
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
print("action observation flattened: ", action_input.shape, observation_input.shape, flattened_observation.shape)

actor = build_actor(env, nb_actions)
critic = build_critic(action_input, flattened_observation, observation_input)
processor = MultiInputProcessor(nb_inputs=5)
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3, processor=processor)
agent.compile(Adam(learning_rate=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=200)
