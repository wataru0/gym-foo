import gym
import mujoco_py
import gym_foo
# from gym import wrappers

env = gym.make('foo-v0')
# env = wrappers.Monitor(env, './mj-video/')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())