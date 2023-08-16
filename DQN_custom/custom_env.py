import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
import tensorflow as tf

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
        print("[state]: ", self.state)
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
        self.state = 0#38 + random.randint(-5,5)
        self.shower_length = 60 
        return self.state