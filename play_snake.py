from snake import Snake_Game
from snake_ai import Snake_AI
from snake_ai import format_state
from snake_ai import State
import numpy as np
import matplotlib.pyplot as plt

grid_x = 10
grid_y = 10

np.random.seed()

def run():
    snake_ai = Snake_AI()
    
    epsilon_decay = 0.01
    min_epsilon = 0.1

    for count in range(0, 100):
        game = Snake_Game(show=True)
        alive = True

        while(alive):
            game_state = game.get_state()
            state_old = State(game_state[0], game_state[1])
            score_old = game_state[2]

            epsilon = 1 - epsilon_decay * count
            if(epsilon < min_epsilon):
                epsilon = min_epsilon

            if(np.random.random() < epsilon):
                # pick a random move
                move = np.random.randint(0, 3)
                print(move)
            else:
                move = snake_ai.get_best_action(state_old)

            alive = game.mvstep(move)
            game_state = game.get_state()
            state = State(game_state[0], game_state[1])
            score = game_state[2]

            reward = get_reward(score_old, score, state_old, state, alive)
            end = reward == 1 or reward == -1
            # to model snake as fully deterministic, both losing and collecting
            # a pellet are considered 'end' conditions (eg. reward != 0)
            snake_ai.observe_reward(state_old, state, move, reward, end)

            if(not alive):
                print('Game:' + str(count) + ' Score:' + str(score))

        snake_ai.experience_replay()
    snake_ai._model.save('./model')
    plt.plot([h.history['loss'] for h in snake_ai.history])
    plt.show()
    plt.savefig('model_loss.png')

def get_reward(score_old, score_new, state, state_new, alive):
    if(score_new > score_old):
        return 1
    if(not alive):
        return -1

    # x1 = np.abs(state_new.body[0][0] - state_new.pellet[0])
    # y1 = np.abs(state_new.body[0][1] - state_new.pellet[1])
    # x2 = np.abs(state.body[0][0] - state.pellet[0])
    # y2 = np.abs(state.body[0][1] - state.pellet[1])

    # if(np.hypot(x1, y1) < np.hypot(x2, y2)):
        # return 0.1
    return 0

run()
