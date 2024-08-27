import gym
import mujoco_py

# Create a MuJoCo environment
env = gym.make('HalfCheetah-v2', render_mode='human')  # or 'Ant-v4', 'Hopper-v4', etc.
env.reset()

# Run a few steps in the environment
for _ in range(100):
    env.render()
    action = env.action_space.sample()  # Take random actions
    env.step(action)

env.close()
