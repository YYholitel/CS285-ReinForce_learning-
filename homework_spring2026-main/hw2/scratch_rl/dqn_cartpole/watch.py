import gymnasium as gym
import torch

from q_network import QNetwork


env = gym.make("CartPole-v1", render_mode="human")

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = "cuda" if torch.cuda.is_available() else "cpu"

q_network = QNetwork(obs_dim, action_dim).to(device)
q_network.load_state_dict(torch.load("dqn_cartpole.pt", map_location=device))
q_network.eval()

for episode in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = q_network(obs_tensor)
            action = q_values.argmax(dim=1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"episode {episode + 1}, return = {total_reward}")

env.close()