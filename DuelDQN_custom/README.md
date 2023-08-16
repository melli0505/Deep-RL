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
- model name : dueling

Also, please comment `dueling_dqn`, in `choose_action` function, line 38-40 not to choose random action during the evaluation.

## Warning
I set the effect of reward on training step 10 times larger than normal one. You can check it out in `dueling_dqn.py`, line 65. 

If your model hardly depends on the reward function, this might be the reason. 