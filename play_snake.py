from snake import Snake_Game
from snake_ai import Snake_AI
from snake_ai import format_state
import numpy as np
import random

random.seed()

grid_x = 5
grid_y = 5

def run():
    snake_ai = Snake_AI()
    
    for count in range(0, 100):
        game = Snake_Game()
        alive = True
        while(alive):

            state_old = game.get_state()

            rand_chance = 0.40 - 0.005 * count
            if(random.random() < rand_chance):
                # pick a random move
                move = np.array([0, 0, 0, 0], dtype=np.uint8)
                move[random.randint(0, 3)] = 1
            else:
                move = snake_ai.model.predict(np.array([format_state(game.get_state())]))

            alive = game.mvstep(np.argmax(move))
            state = game.get_state()

            snake_ai.train_short_memory(state_old, move, 
                    snake_ai.get_reward(state_old, state, not alive),
                    state, not alive)

            if(not alive):
                print('Game:' + str(count) + ' Score:' + str(state[3]))
        
        snake_ai.replay_train()

    snake_ai.model.save('snake_model')

run()
