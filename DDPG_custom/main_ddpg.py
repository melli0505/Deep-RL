import gym
import numpy as np
from ddpg_tf2 import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 250

    figure_file = 'plots/pendulum.png'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_games):
        episode_total = 0
        observation = env.reset()
        done = False
        score = 0
        while not done:
            episode_total += 1
            if score == 0:
                action = agent.choose_action(observation[0], evaluate)
            else:
                action = agent.choose_action(observation, evaluate)
            observation_, reward, done, _, info = env.step(action)
            score += reward
            agent.remember(observation[0], action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
            if episode_total % 100 == 0:
                print("episode: ", i, " | step: ", episode_total, " | total score: ", score)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

