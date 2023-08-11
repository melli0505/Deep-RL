from gym import Env
from gym.spaces import Box, Discrete
import random
import numpy as np

class ENV(Env):
    def __init__(self):
        # self.action_space = Discrete(5, start=-2)
        self.action_space = Box(low=-3, high=3, shape=(1,), dtype=np.int8)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=np.int8)
        self.state = 23 + random.randint(-5,5)
        self.shower_length = 100

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
        # print("+ env action: ", action)
        self.state += action[0] * 3
        self.shower_length -= 1 
        
        # Calculating the reward
        if self.state > 30:
            reward = -5
        elif 30 >= self.state > 25:
            reward = -3
        elif self.state >=20 and self.state <=25: 
            reward = 5
        elif 10 < self.state < 20:
            reward = -3
        else:
            reward = -5
        
        # Checking if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Setting the placeholder for info
        info = {}
        
        # Returning the step information
        return self.get_obs(), reward, done, info
    
    def reset(self):
        self.state = 23 + random.randint(-5,5)
        self.shower_length = 100 
        return self.get_obs()
    
    def get_obs(self):
        return np.array([self.state], dtype=int)
