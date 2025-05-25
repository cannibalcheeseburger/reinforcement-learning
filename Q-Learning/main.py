import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter  

returns = []

random.seed(96)
mapsize = 8
env_args = dict(map_name=f"{mapsize}x{mapsize}", is_slippery=False)
outfolder = '..'
env = gym.make('FrozenLake-v1', **env_args, render_mode="ansi")
state_space = env.observation_space.n
action_space = env.action_space.n
Q = np.zeros((state_space, action_space))
writer = SummaryWriter(logdir=f'{outfolder}/runs/QLearning_{mapsize}x{mapsize}')  

alpha = 0.8
gamma = 0.95
epsilon = 0.2
episodes = 300

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
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
            # goal_row, goal_col = divmod(env.unwrapped.observation_space.n - 1, env.unwrapped.desc.shape[1])
            # current_row, current_col = divmod(new_state, env.unwrapped.desc.shape[1])
            # distance = abs(goal_row - current_row) + abs(goal_col - current_col)
            # reward += 0.5 / (distance + 1)  
            reward-=1
        Q[state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action])
        state = new_state
        total_reward += reward
    returns.append(total_reward)
    writer.add_scalar('Return/Episode', total_reward, episode)
    print('Return:',total_reward,'Episode:',episode)
env.close()
writer.close()  


test_env = gym.make('FrozenLake-v1', **env_args, render_mode="human")  
state = test_env.reset()[0]  
done = False
while not done:
    action = np.argmax(Q[state])
    state, reward, done, truncated, info = test_env.step(action)
    test_env.render()  
test_env.close()  

# plt.figure(figsize=(10,6))
# plt.plot(returns, label='Return per Episode', color='cyan')
# plt.xlabel('Episode')
# plt.ylabel('Return')
# plt.title('Return vs Episode')
# plt.legend()
# plt.grid(True)
# # plt.savefig('output/Return_Episode.png')


# q_values_grid = np.max(Q, axis=1).reshape((8, 8))

# plt.figure(figsize=(6, 6))
# plt.imshow(q_values_grid, cmap='coolwarm', interpolation='nearest')
# plt.colorbar(label='Q-value')
# plt.title('Learned Q-values for each state')
# plt.grid(True)

# for i in range(8):
#     for j in range(8):
#         plt.text(j, i, f'{q_values_grid[i, j]:.2f}', ha='center', va='center', color='black')

# # plt.savefig('output/Q_value.png')

# print("Run 'tensorboard --logdir runs' to view performance graphs!")
