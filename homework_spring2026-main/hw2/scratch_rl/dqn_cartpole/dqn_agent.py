# this agent is a simple DQN agent that uses a neural network to approximate the Q-function. It uses an epsilon-greedy policy for exploration and updates its Q-values using the Bellman equation.

import random 
import numpy as np 
import torch 
from torch.nn import functional as F
from q_network import QNetwork

class DQNAgent:
    def __init__(self,obs_dim,action_dim,hidden_dim=64,lr=1e-3,gamma=0.99,epsilon=0.1,device='cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        
        self.q_network = QNetwork(obs_dim,action_dim,hidden_dim).to(device)
        self.target_q_network = QNetwork(obs_dim,action_dim,hidden_dim).to(device)
        
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(),lr=lr)
        
    def select_action(self,obs,epsilons):
      if random.random() < epsilons: 
        return random.randrange(0,self.action_dim)  
      obs_tensor = torch.tensor(obs,dtype=torch.float32).unsqueeze(0).to(self.device)
      
      with torch.no_grad():
        q_values = self.q_network(obs_tensor)
      action =q_values.argmax(dim =1 ).item()
      return action
    #  train on a batch of data from the replay buffer
    def update(self, batch):    
      obs = torch.tensor(batch[0],dtype=torch.float32).to(self.device)
      actions = torch.tensor(batch[1],dtype=torch.int64).unsqueeze(1).to(self.device)
      rewards = torch.tensor(batch[2],dtype=torch.float32).to(self.device)
      next_obs = torch.tensor(batch[3],dtype=torch.float32).to(self.device)
      dones = torch.tensor(batch[4],dtype=torch.float32).to(self.device)
      
      q_values = self.q_network(obs)
      q_sa =q_values.gather(1,actions).squeeze(1)
      with torch.no_grad():   
        next_q_values = self.target_q_network(next_obs).max(dim=1)[0]
        target = rewards  + self.gamma* next_q_values*(1-dones)
      loss = F.mse_loss(q_sa,target)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      return loss.item()


    def sync_target(self):
      self.target_q_network.load_state_dict(self.q_network.state_dict()) 
      