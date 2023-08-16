# Deep Reinforcement Learning with Custom Environment
This repository includes various Deep Reinforcement learning model training with a custom environment.
- I created a custom model for my case using the gym library and modified some model structures and training sequences.
- I haven't implemented complete models. Each model structure and wrapper have their own repositories or references, which I have mentioned in each model's README.

## Custom Environment
- The Custom Environment simulates a Virtual Temperature Control System. Each action represents a change in temperature after performing an virtual control action (not implemented, you can insert your own real-world simulation environment).
- The target temperature in this system ranges between 20-25. I've set different rewards based on differences between the state (temperature) and the target temperature. You can adjust the reward system to match your own system.
- I haven't implemented the render and close functions in the environment object. If needed, you can refer to examples in the original gym repository (https://github.com/openai/gym/tree/master/gym/envs).

## DQN with keras-RL Agent
Apply keras-rl DQN Agent(https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py) to custom virtual Temperature control environment.
- Reference - https://www.section.io/engineering-education/building-a-reinforcement-learning-environment-using-openai-gym/

### Theory(in korean)
- Markov Decision Process & Q-Learning(https://dnai-deny.tistory.com/80)
- DQN (https://dnai-deny.tistory.com/81)

## DDPG with tensorflow
Deformed DDPG Agent and Actor-Critic models that implemented by ***@philtabor***. Original Repository and Code is in https://github.com/philtabor/Youtube-Code-Repository.

## Dueling DQN with tensorflow
Deformed Dueling DQN models that implemented by ***@philtabor***. Original Repository and Code is in https://github.com/philtabor/Youtube-Code-Repository.

I set the effect of reward on training step 3 times larger than normal one. You can check it out in `dueling_dqn.py`, line 65. 

For more information, see `DuelDQN_custom/README.md`

## Dueling Double DQN with tensorflow
Deformed Dueling DQN models that implemented by ***@philtabor***. Original Repository and Code is in https://github.com/philtabor/Youtube-Code-Repository.

I set the effect of reward on training step 10 times larger than normal one. You can check it out in `dddqn.py`, line 59 - 61. 

For more information, see `DDDQN_custom/README.md`
