import gym
import numpy as np
import matplotlib.pyplot as plt
import statistics


def run_episode(env, weights):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = agent(observation, weights)
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


def agent(observation, weights):
    if np.matmul(weights, observation) >= 0:
        action = 1
    else:
        # action = -1 doesnt work and raises Error
        action = 0
    return action


def random_search(env):
    bestweights = None
    bestreward = 0
    num_episodes = 0
    for _ in range(10000):
        # Initialize the weights randomly between [-1, 1]
        weights = np.random.rand(4) * 2 - 1
        reward = run_episode(env, weights)
        num_episodes += 1
        if reward > bestreward:
            bestreward = reward
            bestweights = weights
            # stop at 200 timesteps
            if reward == 200:
                break
    return num_episodes


def main():
    env = gym.make('CartPole-v0')
    num_episodes = []
    for _ in range(1000):
        num_episodes.append(random_search(env))
    env.close()
    print('The Average Number Of Episodes: %.4f' % statistics.mean(num_episodes))
    # plot accuracy
    plt.clf()
    plt.hist(num_episodes, bins=80)
    # Add title and axis names
    plt.title('Histogram Of Random Search')
    plt.xlabel('Episodes required to reach 200')
    plt.ylabel('Number Of Searches')
    plt.xlim(0)
    plt.show()


if __name__ == "__main__":
    main()
