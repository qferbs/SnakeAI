import tensorflow as tf
from tensorflow import keras
import numpy as np
from snake import Snake_Game
import random

class Snake_AI:

    grid_x = 10
    grid_y = 10
    gamma = 0.9

    def __init__(self):
        self.model = self._get_model()
        self.memory = []
        self.short_memory = []

    ''' trains the network based on immediate state change and reward and saves
        input for later replay training.
        params:
            state: game information in matrix required for NN input.
            move: array of length 4 with each index representing a move.
                  the index with max value is the move that was performed.
            reward: value of reward given for this move.
            next_state: state achieved by move.
            end: boolean which is true if the game ended from this move. '''
    def train_short_memory(self, state, move, reward, next_state, end):
        self.memory.append((state, move, reward, next_state, end))
        target_val = reward
        if not end:
            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
        target_val_new = self.model.predict(np.array([state]))
        target_val_new[0][np.argmax(move)] = target_val
        self.model.fit(np.array([state]), target_val_new, epochs=1, verbose=0)

    ''' trains the NN on earlier positions, allowing rewards to propogate
        throughout the model. '''
    def replay_train(self):
        if(len(self.memory) > 1000):
            batch = random.sample(self.memory, 1000)
        else:
            batch = self.memory
        for state, move, reward, next_state, end in batch:
            target_val = reward
            if not end:
                target = reward + self.gamma * np.amax(self.model.predict(
                    np.array([next_state]))[0])
            target_val_new = self.model.predict(np.array([state]))
            target_val_new[0][np.argmax(move)] = target_val
            self.model.fit(np.array([state]), target_val_new, epochs=1, verbose=0)

    def get_reward(self, state_old, state, end):
        reward = 0
        if(state[3] > state_old[3]):
            reward += 10
        if(end):
            reward -= 100
        x1 = np.abs(state[0][0] - state[2][0])
        x2 = np.abs(state[0][1] - state[2][1])
        x3 = np.abs(state_old[0][0] - state_old[2][0])
        x4 = np.abs(state_old[0][1] - state_old[2][1])
        if(np.hypot(x1, x2) < np.hypot(x3, x4)):
            reward += 1
        return reward
            

    ''' Gets neural network's model. the input is three matrices of binary map data
        (position of head, position of body, and position of pellet) concatenated in
        the x-axis, and the output an array 4 representing each possible move '''
    def _get_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(self.grid_x*3, self.grid_y)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(4, activation='softmax'),
        ])
        
        model.compile(optimizer='adam',
                loss='mse')

        return model

