# replay buffer record each trajectory and use a ramdom sample to train the model 

import random 
from collections import deque

import numpy as np 
# this is implemented by using deque : FIFO queue 
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, obs,action,reward,next_obs,done):
        self.buffer.append((obs,action,reward,next_obs,done))
        
    def sample(self,batch_size): # randomly pick a batch of data from the buffer
        batch = random.sample(self.buffer,batch_size)
        obs,action,reward,next_obs,done = map(np.array,zip(*batch))# turn the list of tuples into a tuple of list and then convert them into numpy array
        return obs,action,reward,next_obs,done

    def __len__(self):  
        return len(self.buffer)
