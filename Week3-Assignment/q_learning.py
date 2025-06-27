import numpy as np
import gym
import imageio
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="rgb_array")

n_states = env.observation_space.n
n_actions = env.action_space.n

def save_video(frames, filename="frozenlake_qlearning.mp4"):
    os.makedirs("videos", exist_ok=True)
    path = f"videos/{filename}"
    with imageio.get_writer(path, fps=3, codec='libx264') as writer:
        for frame in frames:
            writer.append_data(frame)

# -------------------
# Q-Learning
# -------------------
def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
    Q = np.zeros((n_states, n_actions))
    rewards = []

    for ep in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, _, _ = env.step(action)
            best_next = np.argmax(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next] - Q[state, action])
            state = next_state
            total_reward += reward

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards.append(total_reward)
        
        if total_reward > 0:
            print(f"✔️ Reached goal in episode {ep}")

    return Q, rewards

# -------------------
# Evaluation & Video
# -------------------
def evaluate_policy(Q, name="qlearning"):
    state = env.reset()[0]
    frames = [env.render()]
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(Q[state])
        state, reward, done, _, _ = env.step(action)
        frames.append(env.render())
        total_reward += reward

    save_video(frames, f"{name}.mp4")
    return total_reward


# Record Videos BEFORE Training
# evaluate_policy(np.zeros((n_states, n_actions)), "qlearning_untrained")


# Train and Plot

q_Q, q_rewards = q_learning(env)

def smooth(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode='valid')

plt.plot(smooth(q_rewards), label="Q-Learning (Smoothed)")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("FrozenLake-v1: Reward per Episode")
plt.legend()
plt.grid()
plt.show()

# Record Videos
evaluate_policy(q_Q, "qlearning")


