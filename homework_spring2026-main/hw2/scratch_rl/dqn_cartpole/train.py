import gymnasium as gym
import numpy as np
import torch

from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

env_name = "CartPole-v1"

episodes = 300
batch_size = 64
buffer_size = 50000
learning_starts = 1000

gamma = 0.99
lr = 1e-3

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 20000

target_update_freq = 500

# initialize environment
device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make(env_name)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(
    obs_dim=obs_dim,
    action_dim=action_dim,
    gamma=gamma,
    lr=lr,
    device=device,
)

replay_buffer = ReplayBuffer(buffer_size)

def get_epsilon(step):
    progress = min(1.0, step / epsilon_decay_steps)
    return epsilon_start + progress * (epsilon_end - epsilon_start)
  
# training loop
returns = []
global_step = 0 

for episode in range(1,episodes+1):
    obs, _ = env.reset()
    done = False
    episode_return = 0
    
    while not done:
#sample a batch of data from the replay buffer
        epsilon = get_epsilon(global_step)
        action = agent.select_action(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        replay_buffer.add(obs, action, reward, next_obs, done)
        
        obs = next_obs
        episode_return += reward
        global_step += 1
# train the agent if enough data is collected in the replay buffer
        if len(replay_buffer) > learning_starts:
            batch = replay_buffer.sample(batch_size)
            loss = agent.update(batch)
            
            if global_step % target_update_freq == 0:
                agent.sync_target()
    
    returns.append(episode_return)
    
    if episode % 10 == 0:
        avg_return = np.mean(returns[-10:])
        print(f"Episode: {episode}, Average Return: {avg_return:.2f}, Epsilon: {epsilon:.2f}")
        
        
torch.save(agent.q_network.state_dict(), "dqn_cartpole.pt")
env.close()
