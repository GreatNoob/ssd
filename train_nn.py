import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNLSTMPolicy(nn.Module):
    def __init__(self, action_size, input_shape=(3, 64, 64), hidden_size=256):
        super(CNNLSTMPolicy, self).__init__()

        C, H, W = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.feat_dim = 64 * 3 * 3

        self.lstm = nn.LSTM(self.feat_dim, hidden_size, batch_first=True)

        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_size=1, device='cpu'):
        return (torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device))
    
    def to_device(self, device):
        self.lstm.to(device)

    def forward(self, obs, h):
        batch_size, seq_len, C, H, W = obs.size()

        obs = obs.view(batch_size * seq_len, C, H, W)
        x = self.features(obs) 
        x = x.view(batch_size, seq_len, -1) 

        lstm_out, h = self.lstm(x, h)  
        out = lstm_out[:, -1, :]  
        
        logits = self.policy_head(out)
        value = self.value_head(out)

        return logits, value, h

    def act(self, obs, h):

        obs = obs.unsqueeze(0).unsqueeze(0)
        logits, value, h = self.forward(obs, h)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(), h

def compute_returns_and_advantages(rewards, values, gamma=0.99):
    # from stable_baselines3
    returns = []
    R = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        R = r + gamma * R
        returns.append(R)
    returns = torch.tensor(returns[::-1], dtype=torch.float32).to(device)
    advantages = returns - values.detach()
    return returns, advantages


def simulate_episode(env, policy):
    obs, info = env.reset()
    
    h_states = {agent_id: policy.init_hidden(batch_size=1, device=device) for agent_id in obs.keys()}

    # agent_actions = {agent_id: [] for agent_id in obs.keys()}
    agent_log_probs = {agent_id: [] for agent_id in obs.keys()}
    agent_values = {agent_id: [] for agent_id in obs.keys()}
    agent_rewards = {agent_id: [] for agent_id in obs.keys()}

    for _ in tqdm(range(1000)): 
        actions_dict = {}
        for agent_id, agent_obs_ in obs.items():
            agent_obs = agent_obs_
            agent_obs_tensor = torch.tensor(agent_obs.copy(), dtype=torch.float32, device=device)
            agent_obs_tensor = agent_obs_tensor.permute(2, 0, 1)

            action, log_prob, value, h_states[agent_id] = policy.act(agent_obs_tensor, h_states[agent_id])

            actions_dict[agent_id] = action
            agent_log_probs[agent_id].append(log_prob)
            agent_values[agent_id].append(value)

        obs, rewards, dones, tunc, infos = env.step(actions_dict)

        for agent_id, r in rewards.items():
            agent_rewards[agent_id].append(r)

    all_log_probs = []
    all_advantages = []
    all_values = []
    all_returns = []

    for agent_id in obs.keys():
        values_tensor = torch.stack(agent_values[agent_id]) 
        returns, advantages = compute_returns_and_advantages(agent_rewards[agent_id], values_tensor)

        log_probs_tensor = torch.stack(agent_log_probs[agent_id]) 
        all_log_probs.append(log_probs_tensor)
        all_advantages.append(advantages)
        all_values.append(values_tensor)
        all_returns.append(returns)

    return all_log_probs, all_values, all_returns, all_advantages, agent_rewards

    
def train(env, num_agents, action_size, input_shape=(3,64,64), hidden_size=256, num_episodes=1000, lr=1e-3, gamma=0.99):
    policy = CNNLSTMPolicy(action_size, input_shape=input_shape, hidden_size=hidden_size).to(device)
    policy.train()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(num_episodes):
        optimizer.zero_grad()
        all_log_probs, all_values, all_returns, all_advantages, agent_rewards = simulate_episode(env, policy)

        all_log_probs = torch.cat(all_log_probs)
        all_advantages = torch.cat(all_advantages)
        all_values = torch.cat(all_values)
        all_returns = torch.cat(all_returns)

        policy_loss = -(all_log_probs * all_advantages).mean()
        value_loss = (all_returns - all_values).pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        loss.backward()
        optimizer.step()

        avg_reward = np.mean([sum(r) for r in agent_rewards.values()])
        print(f"episode {ep}, loss: {loss.item():.4f}, avg reward per agent: {avg_reward:.2f}")

    env.close()
    return policy

from social_dilemmas.envs import cleanup, agent

if __name__ == "__main__":
    num_agents = 5
    env = cleanup.CleanupEnv(num_agents=num_agents)

    policy = train(env, num_agents, 8, env.observation_space("").shape)
    torch.save(policy.state_dict(), "policy_lstm_nn.pth")

    # policy = CNNLSTMMultiAgentPolicy(9, input_shape=(3,15,15), hidden_size=256)