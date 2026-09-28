import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

try:
    for episode in range(50):
        state, info = env.reset()
        total_reward = 0

        while True:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"任务终止：{terminated}，达到步数上限：{truncated}")
                break

        print(f"第 {episode + 1} 局，总奖励：{total_reward}")
finally:
    env.close()