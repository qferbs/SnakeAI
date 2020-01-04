from snake import Snake_Game
from snake_ai import Snake_AI
import numpy as np
import random

random.seed()

grid_x = 10
grid_y = 10

def format_state(state):
    out = np.zeros((3*grid_x, grid_y))

    head = state[0]
    body = state[1]
    pellet = state[2]

    out[head[0]][head[1]] = 1
    for body_part in body:
        out[grid_x + body_part[0]][body_part[1]] = 1
    out[grid_x*2 + pellet[0]][pellet[1]] = 1

    return out

def run():
    snake_ai = Snake_AI()
    
    for count in range(0, 100):
        game = Snake_Game()
        alive = True
        while(alive):

            state_old = game.get_state()

            rand_chance = 0.40
            if(random.random() < rand_chance):
                # pick a random move
                move = [0, 0, 0, 0]
                move[random.randint(0, 3)] = 1
            else:
                move = snake_ai.model.predict(np.array([format_state(game.get_state())]))

            rand_chance -= 0.005
            
            alive = game.mvstep(np.argmax(move))
            state = game.get_state()

            snake_ai.train_short_memory(format_state(state_old), move, 
                    snake_ai.get_reward(state_old, state, not alive),
                    format_state(state), not alive)

            if(not alive):
                print('Game:' + str(count) + ' Score:' + str(state[3]))
        
        snake_ai.replay_train()

run()
