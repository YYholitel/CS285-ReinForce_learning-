# this network  is used to approximate the Q function for the cartpole environment. It takes in the state of the environment and outputs the Q values for each action.

import torch 
from torch import nn 

class QNetwork(nn.Module):
  def __init__(self,obs_dim,action_dim,hidden_dim=64):
    super(QNetwork,self).__init__()
    
    self.net = nn.Sequential(
      nn.Linear(obs_dim,hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim,hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim,action_dim)
    )
    
  def forward(self,obs):
      return self.net(obs)
    
    