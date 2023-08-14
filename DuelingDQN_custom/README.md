# Dueling DQN
Deformed Dueling DQN Agent and Actor-Critic models that implemented by ***@philtabor***.
Original Repository and Code is in https://github.com/philtabor/Youtube-Code-Repository .

## Custom Environment
- `custom_env.py`
- This virtual environment is temperature control system. 
- Temperature needs to be in range 20~25'C, so reward set based on it.
- In this example, state shape is (1, ), so I do modify model structure. You can check in `networks.py` 

## Evaluate with pretrained weight
There is pretrained weight after training 180 episode in `models/`. You can test with this, set input as following comments in `main.py`.
- Load checkpoint : True
- Training resume : False
- model name : dueling

If you wanna train with your own environment setting, you need to modify line 42 in `main.py`.

## Warning
I set the effect of reward on training step 10 times larger than normal one. You can check it out in `dueling_dqn.py`, line 59 - 61. 

If your model hardly depends on the reward function, this might be the reason. 