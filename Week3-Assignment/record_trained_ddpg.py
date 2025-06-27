import gym
import torch
import numpy as np
import imageio
import os

# ----- Load Trained DDPG Actor Model -----
class Actor(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, act_dim),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.net(x) * self.max_action

# ----- Load Environment -----
env = gym.make('Pendulum-v1', render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Load the trained actor model
actor = Actor(obs_dim, act_dim, max_action)
actor.load_state_dict(torch.load("ddpg_actor.pth"))
actor.eval()

# ----- Save Video Function -----
def save_video(frames, filename="ddpg_pendulum.mp4"):
    os.makedirs("videos", exist_ok=True)
    path = f"videos/{filename}"
    with imageio.get_writer(path, fps=30, codec='libx264') as writer:
        for frame in frames:
            writer.append_data(frame)

# ----- Record DDPG Agent -----
def record_video(actor, max_frames=4800):
    obs = env.reset()[0]
    frames = []
    frame_count = 0
    done = False

    while not done and frame_count < max_frames:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = actor(obs_tensor).numpy()
        obs, _, done, _, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        frame_count += 1

    save_video(frames, "ddpg_pendulum.mp4")
    print("✅ Trained DDPG video saved in 'videos/ddpg_pendulum.mp4'")

record_video(actor)
