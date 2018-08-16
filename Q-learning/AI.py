import tensorflow as tf
import cv2
import numpy as np
from collections import deque
import time, os
from Game import Map

# Pre-trained model
FILENAME = "model.npy"

# Game setting
N_SQUARE_ROW = 10
SQUARE = 30
MARGIN = 5
WINDOW = SQUARE * N_SQUARE_ROW + (MARGIN * (N_SQUARE_ROW + 1))

# Hyper parameters
ACTIONS = 8
N_EPISODES = 2001


class Agent():
    def __init__(self, model):
        self.q_table = np.zeros((WINDOW, WINDOW) + (ACTIONS, ), )
        self.epsilon = 0.7
        self.gamma = 0.99
        self.alpha = 0.01
        self.actions = ACTIONS
        if model:
            self.load_model()

    def pick_action(self, state):
        if np.random.rand() > self.epsilon or (not np.any(self.q_table[state])):
            action = np.random.randint(low=0, high=self.actions)
        else:
            action = np.argmax(self.q_table[state])
            
        arr_action = np.zeros(ACTIONS)
        arr_action[action] = 1
        return arr_action
    
    def test(self, state):
        action = np.argmax(self.q_table[state])
        arr_action = np.zeros(ACTIONS)
        arr_action[action] = 1
        return arr_action
    
    def take_step(self, env, action):
        new_state, reward, win = env.take_action(action)
        return new_state, reward, win

    def update_q_table(self, s, a, n_s, r):
        if r == -1.04:
            self.q_table[s + (np.argmax(a), )] += self.alpha * \
                (r - self.q_table[s + (np.argmax(a), )])
        else:
            self.q_table[s + (np.argmax(a), )] += self.alpha * (r + self.gamma * np.amax(
                self.q_table[n_s]) - self.q_table[s + (np.argmax(a), )])

    def load_model(self):
        loaded_q_table = np.load(FILENAME)
        if loaded_q_table.shape == self.q_table.shape:
            self.q_table = loaded_q_table

    def save_model(self):
        np.save(FILENAME, self.q_table)


class Environment():
    def __init__(self):
        self.agent_state = (0, 0)
        self.game = None
        
    def setup(self):
        self.game = Map(WINDOW, SQUARE, N_SQUARE_ROW, MARGIN)
        self.game.getPresentFrame()

    def take_action(self, action):
        reward, state = self.game.getNextAction(action)
        self.agent_state = state
        win = True if reward == 0.96 else False
        return state, reward, win
    
    def get_state(self):
        return self.agent_state
    
model = None
if os.path.isfile(FILENAME):
    model = FILENAME

agent = Agent(model)
env = Environment()

if not model:
    for i in range(N_EPISODES):
        win = False
        env.setup()
        while not win:
            s = env.get_state()
            a = agent.pick_action(s)
            next_s, r, win = agent.take_step(env, a)
            agent.update_q_table(s, a, next_s, r)
        print("Episode:", i)
        if i % 500 == 0:
            agent.save_model()

print("Press enter to test your model")
input()
win = False
env.setup()
while not win:
    time.sleep(0.5)
    s = env.get_state()
    a = agent.test(s)
    next_s, r, win = agent.take_step(env, a)
    agent.update_q_table(s, a, next_s, r)
