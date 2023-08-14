from gym import Env
from gym.spaces import Box, Discrete
import random
import numpy as np

class ENV(Env):
    def __init__(self):
        self.action_space = [i for i in range(-2, 3)]
        self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=np.int8)
        self.state = 23 + random.randint(-10,10)
        self.shower_length = 200

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
        self.state += self.action_space[action] # [0] * 3
        self.shower_length -= 1 
        
        if self.state >= 20 and self.state <= 25:
            reward = +100
        else:
            reward = -100

        # Checking if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        info = {}
        
        return self.get_obs(), reward, done, info
    
    def reset(self):
        self.state = np.random.choice([-20, 0, 20, 40, 60, 80])
        self.shower_length = 200 
        return self.get_obs()
    
    def set_state(self, s):
        self.state = s
        return self.get_obs()
    
    def get_obs(self):
        return np.array([self.state], dtype=int)
