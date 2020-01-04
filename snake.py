import numpy as np
import random
from enum import Enum

class Snake_Game:
    grid_size = 10    

    def __init__(self, show=True):
        self.show = show
        random.seed()
        self.snake = [(2,1), (1,1)]
        self.direction = 0
        self.pellet = (0, 0)
        self._new_pellet()
        self.score = 0

        if(show):
            self._show()
    
    ''' Applies one game step, moving the snake in the direction of
        the last input direction to dir() and returns the game state.
        returns: false if game has been lost, true otherwise '''
    def step(self):
        
        head = self._move_snake()

        if(np.all(head == self.snake[1])):
            if(self.direction < 2):
                self.direction += 2
            else:
                self.direction -= 2
            head = self._move_snake()

        if(head[0] < 0 or head[0] >= self.grid_size or (head[1] < 0
                or head[1] >= self.grid_size)
                or np.any([np.all(head == s) for s in self.snake])):
            return False
        elif(head == self.pellet):
            self.score += 1
            self._new_pellet()
            self.snake = np.concatenate(([head], self.snake))
        else:
            self.snake = np.concatenate(([head], self.snake[:(len(self.snake) - 1)]))

        if(self.show):
            self._show()

        return True

    def _move_snake(self):
        if(self.direction == 0):
            head = (self.snake[0][0] + 1, self.snake[0][1])
        elif(self.direction == 1):
            head = (self.snake[0][0], self.snake[0][1] + 1)
        elif(self.direction == 2):
            head = (self.snake[0][0] - 1, self.snake[0][1])
        elif(self.direction == 3):
            head = (self.snake[0][0], self.snake[0][1] - 1)
        else:
            raise ValueError('Invalid Direction:' + str(self.direction))

        return head

    ''' returns the current game state in the form of (head, body, pellet, score),
        where head is a tuple(x, y), body is an array of tuples (x, y), pellet
        is a tuple (x, y), and score is an integer. '''
    def get_state(self):
        return (self.snake[0], self.snake[1:], self.pellet, self.score)

    ''' queues snake to move in the direction given through param dir,
        which corresponds to:
            right: 0
            up: 1
            left: 2
            down: 3 '''
    def move(self, dir):
        self.direction = dir

    def mvstep(self, dir):
        self.move(dir)
        return self.step()

    def _show(self):
        for y in reversed(range(0, self.grid_size)):
            out = ''
            for x in range(0, self.grid_size):
                if(np.any([np.all((x, y) == s) for s in self.snake])):
                    out = out + 's'
                elif((x, y) == self.pellet):
                    out = out + 'p'
                else:
                    out = out + '-'
                
            print(out)

    # places a new pellet, making sure it is not inside snake
    def _new_pellet(self):
        self.pellet = (random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1))
        while(np.any([self.pellet == s for s in self.snake])):
            self.pellet = (random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1))
'''
snake = Snake_Game()
print(snake.get_state())
snake.move(1)
snake.step()
print(snake.get_state())
'''
