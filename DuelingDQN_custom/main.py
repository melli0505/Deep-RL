import time
import numpy as np
from dueling_dqn import Agent
from custom_env import ENV
from network import DuelingDeepQNetwork
import tensorflow as tf

if __name__ == '__main__':
    env = ENV()
    agent = Agent(env=env, lr=1e-3, gamma=0.99, n_actions=5, epsilon=1.0,
                  batch_size=64, input_dims=[1])
    n_games = 3
    ddqn_scores = []
    eps_history = []

    load_checkpoint = bool(input("Load checkpoint? (True/False) : "))
    training_resume = bool(input("Training resume? (True/False) : "))

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = np.random.choice(env.action_space)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        if training_resume: model_name = input("Enter model name under models dir (ex. dueling): ")
        agent.load_models(model_name)
        evaluate = not training_resume
    else:
        evaluate = False


    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        print("initial state: ", observation, end='\t')
        while not done:
            # if you train model, change choose_action's evaluate to False
            action = agent.choose_action(observation, evaluate=True)
            observation_, reward, done, info = env.step(action)
            if abs(observation - 23) > abs(observation_ - 23): 
                if reward != 100: reward = reward + 50 
                else: reward = 100
            if abs(observation - 23) < abs(observation_ - 23): reward -= 50
            # print(f"- action: {action} | state: {observation_} | reward: {reward}")
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            if evaluate:
                agent.learn()
            observation = observation_
        eps_history.append(agent.epsilon)

        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[-100:])
        print('\t| episode: ', i,'\t| score: %.2f' % score, '\t| average score %.2f' % avg_score)
        print(" - last state: ", observation, "\t| reward: ", reward, "\t| action: ", action)
        
        if (i % 5 == 0 and i != 0) or i == n_games - 1:
            if not load_checkpoint:
                agent.save_models('h5_duel' + str(i))
