import time
import gym
import numpy as np
import torch

from agents.pg_agent import PGAgent
from infrastructure import pytorch_util as ptu


# 改成刚才查到的 agent.pt 路径
CHECKPOINT=r"J:\CS285\homework_spring2026-main\hw2\exp\CartPole-v0_cartpole_rtg_na_sd1_20260921_101416\agent.pt"

ptu.init_gpu(use_gpu=False)

# 必须与训练时的网络结构一致
agent = PGAgent(
    ob_dim=4,
    ac_dim=2,
    discrete=True,
    n_layers=2,
    layer_size=64,
    gamma=1.0,
    learning_rate=5e-3,
    use_baseline=False,
    use_reward_to_go=True,
    baseline_learning_rate=5e-3,
    baseline_gradient_steps=5,
    gae_lambda=None,
    normalize_advantages=False,
)

# 加载训练后的参数
state_dict = torch.load(CHECKPOINT, map_location=ptu.device)
agent.load_state_dict(state_dict)
agent.eval()

# human 表示打开动画窗口
env = gym.make("CartPole-v0", render_mode="human")

try:
    for episode in range(10):
        reset_result = env.reset()

        # 同时兼容新旧 Gym API
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result

        total_reward = 0

        while True:
            obs_batch = np.asarray(obs, dtype=np.float32)[None, :]
            obs_tensor = ptu.from_numpy(obs_batch)

            with torch.no_grad():
                action_distribution = agent.actor(obs_tensor)

                # 选择概率最大的动作，演示更稳定
                action = torch.argmax(
                    action_distribution.logits, dim=-1
                ).item()

            step_result = env.step(action)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            total_reward += reward
            time.sleep(0.02)

            if done:
                break

        print(f"第 {episode + 1} 局，总奖励：{total_reward}")
        time.sleep(0.5)

finally:
    env.close()