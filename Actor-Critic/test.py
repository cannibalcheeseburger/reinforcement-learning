import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

learning_rate = 0.0002
gamma = 0.98
n_rollout = 20
map_size = 8
env = gym.make('FrozenLake-v1',map_name = f'{map_size}x{map_size}', is_slippery=False)
state_dim = env.observation_space.n
action_dim = env.action_space.n

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128).to(device)
        self.fc_pi = nn.Linear(128, action_dim).to(device)
        self.fc_v = nn.Linear(128, 1).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.data = []
        
    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        return F.softmax(x, dim=softmax_dim)
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        return self.fc_v(x)
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_lst.append([0.0] if done else [1.0])
            
        return (torch.tensor(s_lst, dtype=torch.float, device=device),
                torch.tensor(a_lst, device=device),
                torch.tensor(r_lst, dtype=torch.float, device=device),
                torch.tensor(s_prime_lst, dtype=torch.float, device=device),
                torch.tensor(done_lst, dtype=torch.float, device=device))
  
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.data = []

def one_hot(state, state_dim):
    vec = np.zeros(state_dim)
    vec[state] = 1.0
    return torch.tensor(vec, dtype=torch.float, device=device)


def test_agent(model):
    env = gym.make('FrozenLake-v1', is_slippery=False, map_name = f'{map_size}x{map_size}', render_mode='human')
    state_dim = env.observation_space.n
    s, _ = env.reset()
    s = one_hot(s, state_dim)
    done = False

    while not done:
        with torch.no_grad():
            prob = model.pi(s)
        a = torch.argmax(prob).item()
        s_prime, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        s = one_hot(s_prime, state_dim)
        env.render()
    env.close()


trained_model = ActorCritic(state_dim, action_dim).to(device)

trained_model.load_state_dict(torch.load("frozenlake_actor_critic.pth", map_location=device))
trained_model.eval()  # Set to evaluation mode if only using for inference
print("Model loaded from frozenlake_actor_critic.pth")


test_agent(trained_model)