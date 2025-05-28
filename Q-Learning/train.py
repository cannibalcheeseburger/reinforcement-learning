import argparse
import gymnasium as gym
import numpy as np
import random
import pickle
from tensorboardX import SummaryWriter

def main(outfolder, mapsize, episodes, is_slippery):
    random.seed(96)
    env_args = dict(map_name=f"{mapsize}x{mapsize}", is_slippery=is_slippery)
    env = gym.make('FrozenLake-v1', **env_args, render_mode="ansi")
    state_space = env.observation_space.n
    action_space = env.action_space.n
    Q = np.zeros((state_space, action_space))
    if is_slippery:
        slip = 'slip'
    else:
        slip = 'no_slip'
    writer = SummaryWriter(logdir=f'{outfolder}/runs/QLearning_{mapsize}x{mapsize}_{slip}')
    alpha = 0.8
    gamma = 0.95
    epsilon = 0.2
    returns = []
    steps = 0
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            steps +=1
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            new_state, reward, done, truncated, info = env.step(action)
            if done and reward == 1:
                reward += 100
            elif done and reward == 0:
                reward -= 50
            else:
                reward -= 1
            Q[state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action])
            state = new_state
            total_reward += reward
        returns.append(total_reward)
        writer.add_scalar('Return',total_reward,steps)
        # writer.add_scalar('Return/Episode', total_reward, episode)
        # print('Return:', total_reward, 'Episode:', episode)

    env.close()
    writer.close()

    # Save Q-table
    q_path = f"Q_table_{mapsize}x{mapsize}.pkl"
    with open(q_path, 'wb') as f:
        pickle.dump(Q, f)
    print(f"Q-table saved to {q_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfolder', type=str, default='.', help='Output folder')
    parser.add_argument('--mapsize', type=int, default=8, help='Size of the FrozenLake map')
    parser.add_argument('--episodes', type=int, default=300, help='Number of training episodes')
    parser.add_argument('--is_slippery', action='store_true', help='Enable slippery terrain')
    args = parser.parse_args()
    main(args.outfolder, args.mapsize, args.episodes, args.is_slippery)
