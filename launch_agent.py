from agent import Agent
import gym
import torch
import random 

agent = Agent(state_size=8, action_size=4, seed=0)
env = gym.make('LunarLander-v2')
env.seed(0)
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for j in range (15):
    for i in range(5):
        state = env.reset()
        for j in range(200):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            if done:
                break 
            