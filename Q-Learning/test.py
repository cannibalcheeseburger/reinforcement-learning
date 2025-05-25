import argparse
import gymnasium as gym
import numpy as np
import pickle

def main(outfolder, mapsize, is_slippery):
    q_path = f"Q_table_{mapsize}x{mapsize}.pkl"
    with open(q_path, 'rb') as f:
        Q = pickle.load(f)

    env_args = dict(map_name=f"{mapsize}x{mapsize}", is_slippery=is_slippery)
    env = gym.make('FrozenLake-v1', **env_args, render_mode="human")
    state = env.reset()[0]
    done = False

    while not done:
        action = np.argmax(Q[state])
        state, reward, done, truncated, info = env.step(action)
        env.render()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfolder', type=str, default='.', help='Output folder')
    parser.add_argument('--mapsize', type=int, default=8, help='Size of the FrozenLake map')
    parser.add_argument('--is_slippery', action='store_true', help='Enable slippery terrain')
    args = parser.parse_args()
    main(args.outfolder, args.mapsize, args.is_slippery)
