from dueling_dqn import Agent
import numpy as np

from custom_env import ENV

if __name__ == '__main__':
    env = ENV()
    n_games = 1
    agent = Agent(gamma=0.5, epsilon=1, lr=1e-3, input_dims=[1], 
                  epsilon_dec=1e-3, mem_size=100000, batch_size=64, eps_end=0.01,
                  fc1_dims=128, fc2_dims=128, replace=100, n_actions=4)

    scores, eps_history = [], []

    load_checkpoint = True

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = np.random.choice(env.action_space)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_model('duel_gamma50_599')

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        print("- initial state: ", observation, end=' ')
        while not done:
            action = agent.choose_action(observation, False)
            observation_, reward, done, info = env.step(action)
            if abs(observation - 23) > abs(observation_ - 23): 
                if reward != 100: reward = reward + 50 
                else: reward = 100
            if abs(observation - 23) < abs(observation_ - 23): reward -= 50
            print("state:", observation_, " | action: ", action, " | reward:", reward)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)

        print("| last state: ", observation)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        if (i % 50 == 0 and i != 0) or i == n_games - 1:
            agent.save_model('duel_gamma50_' + str(i+300))

    # filename='keras_lunar_lander.png'
    # x = [i+1 for i in range(n_games)]
    # plotLearning(x, scores, eps_history, filename)


