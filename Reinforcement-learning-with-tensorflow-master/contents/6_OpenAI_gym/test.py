import gym
from RL_brain import DeepQNetwork

env = gym.make('CartPole-v0')
env = env.unwrapped

observation = env.reset()
print(observation)
print(env.action_space.n)

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)