import numpy as np
import curses

class Snake_Game:
    grid_size = 10

    def __init__(self, show=False, stdscr=None):
        self.show = show
        self.snake = np.array([(4,5), (5,5)], dtype=np.uint8)
        self.direction = 0
        self.pellet = (0, 0)
        self._new_pellet()
        self.score = 0

        if(show):
            self.stdscr = stdscr
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

    def get_state(self):
        return (self.snake, self.pellet, self.score)

    ''' queues snake to move in the direction given through param dir,
        which corresponds to:
            turn left: 0
            go straight: 1
            turn right: 2 '''
    def move(self, move):
        if(move == 0):
            self.direction += 1
        elif(move == 2):
            self.direction -= 1
        if(self.direction > 3):
            self.direction = 0
        elif(self.direction < 0):
            self.direction = 3

    def mvstep(self, dir):
        self.move(dir)
        return self.step()

    def _show(self):
        if(self.stdscr == None):
            print(''.join(['=' for i in range(0, self.grid_size)]))
            for y in reversed(range(0, self.grid_size)):
                out = ''
                for x in range(0, self.grid_size):
                    if(np.all((x, y) == self.snake[0])):
                        out = out + '@'
                    elif(np.any([np.all((x, y) == s) for s in self.snake[1:]])):
                        out = out + '0'
                    elif((x, y) == self.pellet):
                        out = out + '$'
                    else:
                        out = out + ' '
                    
                print(out)
        else:
            self.stdscr.clear()
            self.stdscr.addch(self.grid_size - self.snake[0][1], self.snake[0][0], '@')
            self.stdscr.addch(self.grid_size - self.pellet[1], self.pellet[0], '$')
            for body in self.snake[1:]:
                self.stdscr.addch(self.grid_size - body[1], body[0], '*')
            self.stdscr.refresh()

    # places a new pellet, making sure it is not inside snake
    def _new_pellet(self):
        self.pellet = (np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size))
        i = 0
        while(np.any([self.pellet == tuple(s) for s in self.snake])):
            if(i == 1000):
                import ipdb; ipdb.set_trace()

            self.pellet = (np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size))
            i += 1
'''
snake = Snake_Game()
print(snake.get_state())
snake.move(1)
snake.step()
print(snake.get_state())
'''
