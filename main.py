import pandas as pd
import numpy as np
import gym
import random
import os
from os import system
from time import sleep
from tqdm import tqdm_notebook, tqdm
from IPython.display import clear_output


class QLearningAgent():
    def __init__(self):
        env = gym.make('Taxi-v2')

        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.q_table = np.zeros([self.observation_space.n, self.action_space.n])

        self.a_lr = 0.8
        self.epsilon = 0.95
        self.gamma_df = 0.6

    def update_q_value(self, reward, state, action, new_state):
        current_q = self.q_table[state, action]
        self.q_table[state, action] = current_q + self.a_lr * (reward + self.gamma_df * np.max(self.q_table[new_state]) - current_q)

    def epsilon_greedy(self, state):
        if self.epsilon > np.random.rand():
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def training(self, epochs=5000):
        for i in tqdm(range(epochs)):
            done = False
            state = self.env.reset()

            while not done:
                action = self.epsilon_greedy(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_value(reward, state, action, next_state)
                state = next_state
            if i % 500 == 0:
                self.epsilon *= self.epsilon

    def run_episode(self, stupid_strategy=False):
        done = False
        state = self.env.reset()
        timestep = 0

        while not done:
            # clear_output()
            os.system('cls')
            if stupid_strategy:
                action = self.action_space.sample()
            else:
                action = np.argmax(self.q_table[state])
            state, reward, done, _ = self.env.step(action)
            self.env.render()
            timestep += 1
            print(f'state: {state}, reward: {reward}, maxQ: {np.max(self.q_table[state])}')
            sleep(0.5)
        print('Done')


q_agent = QLearningAgent()
q_agent.training(500)
q_agent.run_episode()
