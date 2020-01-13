from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from snake import Snake_Game

grid_x = 10
grid_y = 10

@dataclass(eq=False, frozen=True)
class State:
    body : List[Tuple[int, int]]
    pellet : Tuple[int, int]

    def __hash__(self):
        return int.from_bytes(np.array(self.body + self.pellet).tobytes(), 'big')

    def __eq__(self, other):
        return np.all(self.body == other.body) and self.pellet == other.pellet

def format_state(state) -> np.ndarray:
    out = np.full((grid_x, grid_y, 3), 0.0)

    head = state.body[0]
    body = state.body[1:]
    pellet = state.pellet

    out[head[0]][head[1]][0] = 1.0
    for body_part in body:
        out[body_part[0]][body_part[1]][1] = 1.0
    out[pellet[0]][pellet[1]][2] = 1.0

    return out

class Snake_AI:

    gamma = 0.9
    batch_size = 1024
    minibatch_size = 32
    outlier_threshold = 0.6
    outlier_epochs = 1
    history = []

    def __init__(self):
        self._model = self._get_model()
        self.memory = set([])

    def get_best_action(self, state : State) -> int:
        move_weights = self._get_q([state])[0]
        maximum = np.amax(move_weights)
        print(move_weights)
        return np.random.choice(np.where(move_weights == maximum)[0])

    def observe_reward(self, state : State, state_new : State, action : int,
           reward : int, end : bool) -> None:
        self.memory.add((state, state_new, action, reward, end))

    def experience_replay(self):
        mem = np.array(list(self.memory))
        outliers = []
        if(len(self.memory) > self.batch_size):
            # take random sample of memory
            batch = mem[np.random.choice(mem.shape[0], self.batch_size, replace=False), :]
        else:
            batch = mem
        np.random.shuffle(batch)
        for minibatch in [batch[i:i+self.minibatch_size] for i in range(0,
                int(batch.shape[0] / self.minibatch_size))]:
            # train on minibatch
            minibatch = minibatch.transpose() 

            states = [format_state(s) for s in minibatch[0]]
            targets = self._get_q(minibatch[0])
            state_new_q = self._get_q(minibatch[1])

            for i in range(0, self.minibatch_size):
                new_q_val = minibatch[3][i] + \
                        self.gamma * minibatch[4][i] * np.amax(state_new_q[i])
                if(np.abs(targets[i][minibatch[2][i]] - new_q_val) > self.outlier_threshold):
                    outliers.append(minibatch.transpose()[i])
                targets[i][minibatch[2][i]] = new_q_val

            # (state, state_new, action, reward, end)
         
            self.history.append(self._model.fit(np.array(states), np.array(targets)))

        if(len(outliers) == 0):
            return
        #train on outliers
        for j in range(0, self.outlier_epochs):
            minibatch = np.array(outliers).transpose() 

            states = [format_state(s) for s in minibatch[0]]
            targets = self._get_q(minibatch[0])
            state_new_q = self._get_q(minibatch[1])
            
            for i in range(0, minibatch.shape[1]):
                targets[i][minibatch[2][i]] = minibatch[3][i] + \
                        self.gamma * minibatch[4][i] * np.amax(state_new_q[i])

            self._model.fit(np.array(states), np.array(targets))

    def _get_q(self, states : List[State]) -> np.ndarray:
        st = np.array([format_state(s) for s in states])
        return self._model.predict(st)

    def _get_model(self) -> keras.Model:
        model = keras.Sequential([
            keras.layers.Conv2D(3, 3, padding='same', input_shape=(grid_x, grid_y, 3)),
            keras.layers.Conv2D(3, 3, padding='same', activation='elu'),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='elu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='elu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='elu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='elu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(3, activation='softsign'),
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(0.01),
                loss='mse')

        return model

