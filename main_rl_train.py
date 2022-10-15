import gym
import random
import numpy as np
import torch
import torch.nn as nn
from agent import Agent


env = gym.make("CartPole-v0")
s = env.reset()

EPSILON_DECAY = 100000
EPSILON_START = 1.0
EPSILON_END = 0.02
TARGET_UPDATE_FREQUENCY = 10

n_epsiode = 5000
n_time_step = 1000

n_state = len(s)
n_action = env.action_space.n
print(n_state, n_action)

agent = Agent(n_input=n_state, n_output=n_action)

total_loss = 0
REWARD_BUFFER = np.empty(shape=n_epsiode)
for episode_i in range(n_epsiode):
    episode_reward = 0
    for step_i in range(n_time_step):
        epsilon = np.interp( episode_i * n_time_step + step_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        # epsilon = 0.1
        # print(epsilon)
        random_sample = random.random()
        
        if random_sample <= epsilon :
            a = env.action_space.sample()
        else:
            a = agent.online_net.act(s) # TODO
        
        s_, r, done, info = env.step(a)
        agent.memo.add_memo(s, a, r, done, s_) # TODO
        s = s_
        episode_reward += r

        if done:
            s = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            break

        if np.mean(REWARD_BUFFER[:episode_i]) >= 100:
        # if step_i % 1000 == 0:
            while True:
                a = agent.online_net.act(s)
                s, r, done, info = env.step(a)
                env.render()

                if done:
                    env.reset()
                    break

        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()

        # Compute targets
        target_q_values = agent.target_net(batch_s_)  #TODO
        max_target_q_values = target_q_values.max(dim = 1, keepdim=True)[0]
        targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values
        # print(targets.shape, targets)

        # Compute q_values
        q_values = agent.online_net(batch_s)  #TODO
        
        a_q_values = torch.gather(input=q_values, dim=1, index = batch_a)

        # Compute loss
        loss = nn.functional.smooth_l1_loss(targets, a_q_values)

        # Gradient descent
        agent.optimizer.zero_grad()   #TODO
        loss.backward()
        total_loss += loss.item()
        # print(loss.item())
        agent.optimizer.step()  #TODO

    
    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict()) #TODO

        print(f"Episode: {episode_i} ")
        print(f"Avg. Reward: {np.mean(REWARD_BUFFER[:episode_i])}")
        print(f"total loss:{total_loss / (episode_i + 1)}")
        # while True:
        #     a = agent.online_net.act(s)
        #     s, r, done, info = env.step(a)
        #     env.render()

        #     if done:
        #         env.reset()
        # print(REWARD_BUFFER[:episode_i])







