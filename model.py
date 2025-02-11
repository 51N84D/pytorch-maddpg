import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action, dim):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.FC1 = nn.Linear(obs_dim, dim)
        self.FC2 = nn.Linear(dim + act_dim, dim)
        self.FC3 = nn.Linear(dim, dim)
        self.FC4 = nn.Linear(dim, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action, dim):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, dim)
        self.FC2 = nn.Linear(dim, dim)
        self.FC3 = nn.Linear(dim, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        return result
