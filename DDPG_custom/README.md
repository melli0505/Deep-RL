# DDPG
Deformed DDPG Agent and Actor-Critic models that implemented by @philtabor.
Original Repository and Code is in https://github.com/philtabor/Youtube-Code-Repository .
| WARNING - DDPG is for continuous action space, but this implementation reflect action as integer - discrete after calculate action. You can find source code in `custom_env.py`, line 30 and 62-64.


## Custom Environment
- `custom_env.py`
This virtual environment is temperature control system.
Temperature needs to be in range 20~25'C, so reward set based on it.
In this example, state shape is (1, ), so I do modify actor - critic model structure. You can check in `networks.py` 

## Evaluate with pretrained weight
There is pretrained weight after training 90 episode in `ddpg/`.
You can test with this, modify these things in `custom_ddpg.py`.
- `load_checkpoint = True`
- `n_games = 1`
- uncomment line 50-51 to monitor selected action and reward in each step.