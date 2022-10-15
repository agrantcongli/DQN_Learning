from os import access
import gym
import random
import numpy as np

"""
Goal: learning action-value function
on-policy 选择action的policy和更新Q表的逻辑是否一致,
sarsa 选择了 a_, 并且执行了 a_
可能学不到最优的解, 易收敛到局部最优, 但是加入探索又会导致降低学习效率, 难以找到最优的policy
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
    
    def update(self, s, a, r, s_, a_, done):
        # 结束状态很重要， 没有结束状态是不会收敛的
        if done:
            y = r
        else:
            # TD target = r(真实值) + GAMMA * q (预估值)
            y = r + self.GAMMA * self.Q_table[s_, a_]
        # TD ERROR
        delta = y - self.Q_table[s, a]
        # 利用TD 更新Q表， 减小TD ERROR
        self.Q_table[s, a] = self.Q_table[s, a] + self.learningRate * delta


agent = Agent(n_states, n_actions)

for episode_i in range(n_episode):
    episode_reward = 0
    # 最一开始的第一步进行随机选择 或者按照Q表选择
    if random.random() <= epsilon:
        a = env.action_space.sample()
    else:
        a = agent.act(s)
    while True:
            # print("a is", a)
        s_, r, done, info = env.step(a)
        episode_reward += r
        a_ = env.action_space.sample() if random.random() <= epsilon  else agent.act(s_)
        agent.update(s, a, r, s_, a_, done)
        # 状态更新, 动作更新
        s = s_
        a = a_

        if done :
            # print("next time")
            # print("done, r is ", r)
            env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            # print(episode_reward)
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

env.reset()
while True:
    a = agent.act(s)
    s, r, done, info = env.step(a)
    env.render()
    if done:
        env.reset()
        break










