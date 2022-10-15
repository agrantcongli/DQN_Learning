from os import access
import gym
import random
import numpy as np

"""
Goal: Learn the optimal action-value function Q* 
off-policy 更新Q表所用到的a, 最终不一定被执行 
           更新选择的是argmax(Q*(s, a)), 执行的a 随机选择或argmax(Q*(s, a))

off-policy则会产生大量的探索的结果来提供选择, 但是收敛速度会很慢, 
优势是更加强大与通用, 能保证产生数据的全面性
"""
env = gym.make("CliffWalking-v0")
s = env.reset()

n_episode = 10000
n_time_step = 1000
TARGET_UPDATE_FREQUENCY = 50
epsilon = 0.1
n_states = env.observation_space.n
n_actions = env.action_space.n
print("states, actions", n_states, n_actions)
REWARD_BUFFER = [-100000] * n_episode 


class Agent:
    def __init__(self, n_state, n_action):
        self.n_state = n_state
        self.n_action = n_action
        self.Q_table = np.random.rand(self.n_state, self.n_action)
        self.GAMMA = 0.99
        self.learningRate = 0.01

    def act(self, s):
        return np.argmax(self.Q_table[s]) 
    
    def update(self, s, a, r, s_, done):
        # 结束状态很重要， 没有结束状态是不会收敛的
        if done:
            y = r
        else:
            # TD target = r(真实值) + GAMMA * q (预估值)
            y = r + self.GAMMA * max(self.Q_table[s_])
        # TD ERROR
        delta = y - self.Q_table[s, a]
        # 利用TD 更新Q表， 减小TD ERROR
        self.Q_table[s, a] = self.Q_table[s, a] + self.learningRate * delta


agent = Agent(n_states, n_actions)

for episode_i in range(n_episode):
    episode_reward = 0
    while True:
        if random.random() <= epsilon:
            a = env.action_space.sample()
        else:
            a = agent.act(s)
        s_, r, done, info = env.step(a)
        episode_reward += r
        agent.update(s, a, r, s_, done)
        s = s_

        if done :
            env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            break
        
        # if np.mean(REWARD_BUFFER[:episode_i]) > -41:
        #     while True:
        #         a = agent.act(s)
        #         s, r, done, info = env.step(a)
        #         env.render()
        #         if done:
        #             env.reset()
        #             break

            # env.render()
    
    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        print(f"更新了{episode_i}轮")
        print("平均损失", np.mean(REWARD_BUFFER[:episode_i]), "最新损失",REWARD_BUFFER[episode_i])
    
    # if episode_i > 2000 and  episode_i % 500 == 1:
env.reset()
while True:
    a = agent.act(s)
    s, r, done, info = env.step(a)
    env.render()
    if done:
        env.reset()
        break










