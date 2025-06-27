import gym
import torch
import numpy as np
import imageio
import os

# ----- Load Trained Actor Model -----
class Actor(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, act_dim)
        )
        self.log_std = torch.nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        mu = self.net(x)
        std = self.log_std.exp()
        return mu, std

# ----- Load Environment -----
env = gym.make('Pendulum-v1', render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Load the actor
actor = Actor(obs_dim, act_dim)
actor.load_state_dict(torch.load("ppo_actor.pth"))
actor.eval()

# ----- Save Video Function -----
def save_video(frames, filename="ppo_pendulum.mp4"):
    os.makedirs("videos", exist_ok=True)
    path = f"videos/{filename}"
    with imageio.get_writer(path, fps=30, codec='libx264') as writer:
        for frame in frames:
            writer.append_data(frame)

# def record_video(actor, max_frames=4800):
#     obs = env.reset()[0]
#     frames = []
#     done = False
#     frame_count = 0

#     while not done and frame_count > max_frames and frame_count < max_frames*2 :
#         obs_tensor = torch.tensor(obs, dtype=torch.float32)
#         mu, std = actor(obs_tensor)
#         action = torch.tanh(mu) * max_action
#         action = action.detach().numpy()

#         obs, _, done, _, _ = env.step(action)
#         frame = env.render()
#         frames.append(frame)
#         frame_count += 1

#     save_video(frames, "ppo_pendulum.mp4")
#     print("✅ Trained PPO video saved in 'videos/ppo_pendulum.mp4'")


def record_video(actor, start_frame=4800, end_frame=9600):
    obs = env.reset()[0]
    frames = []
    done = False
    frame_count = 0

    while not done and frame_count < end_frame:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            # PPO-style: actor returns mu, std
            mu, std = actor(obs_tensor)
            action = torch.tanh(mu) * max_action
            action = action.numpy()

        obs, _, done, _, _ = env.step(action)

        # Only start saving frames after reaching start_frame
        if start_frame <= frame_count < end_frame:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        frame_count += 1

    save_video(frames, "ppo_pendulum_segment.mp4")
    print(f"✅ PPO video segment saved: frames {start_frame}–{end_frame} in 'videos/ppo_pendulum_segment.mp4'")

record_video(actor)
