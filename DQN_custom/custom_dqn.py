import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class ENV(Env):
    def __init__(self):
        self.action_space = Discrete(5, start=-2)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
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
        if self.state >=37 and self.state <=39: 
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
    
    def reset(self):
        self.state = 38 + random.randint(-5,5)
        self.shower_length = 60 
        return self.state


Adam._name = 'AdamOptimizer'

def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn



env = ENV()

states = env.observation_space.shape
actions = env.action_space.n

model = build_model(states, actions)
print(model.summary())

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=200000, visualize=False, verbose=1)

results = dqn.test(env, nb_episodes=150, visualize=False)
print(np.mean(results.history['episode_reward']))

# episodes = 20
# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0
    
#     while not done:
#         action = env.action_space.sample()
#         next_state, reward, done, info = env.step(action)
#         score += reward
#     print(f"Episode: {episode} Score: {score}")dummy_ddpg_test.py