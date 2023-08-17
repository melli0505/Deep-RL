import gym
import numpy as np
from ddpg_tf2 import Agent
from utils import plot_learning_curve
from custom_env import ENV

if __name__ == '__main__':
    env = ENV()
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0], fc1=256, fc2=256)
    # figure_file = 'plots/pendulum.png'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = 1 if input("Load checkpoint? (True/False) : ") == 'True' else 0
    training_resume = 1 if input("Training resume? (True/False) : ") == 'True' else 0

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        model_name = input("Enter model name under models dir (ex. dueling): ")
        agent.load_models(model_name)
        evaluate = False if training_resume == 1 else True
    else:
        evaluate = False

    
    n_games = 200 if evaluate is False else 1


    for i in range(n_games):
        observation = env.reset()
        print("initial state: ", env.reset())
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation[0], action, reward, observation_[0], done)
            if not evaluate:
                agent.learn()
            else:
                print("state:", observation_, " | action: ", action, " | reward:", reward)

            observation = observation_
                
            # print("- reward: ", reward)
            # print("- observation: ", observation_)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
        print(" - last state: ", observation, "\t| reward: ", reward, "\t| power: ", env.get_power_consumption()[0])

        if (i % 20 == 0 and i != 0) or i == n_games - 1:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models("ddpg" + str(i))
        
        

    # if not load_checkpoint:
    #     x = [i+1 for i in range(n_games)]
    #     plot_learning_curve(x, score_history, figure_file)

