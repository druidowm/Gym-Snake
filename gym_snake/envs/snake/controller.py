from gym_snake.envs.snake import Snake
from gym_snake.envs.snake import Grid
import numpy as np
import colorsys



def extract_and_pad(array, y1, y2, x1, x2):
    # Create an array of zeros with the desired shape
    result = np.zeros((x2 - x1, y2 - y1, 3), dtype=array.dtype)
    
    # Calculate the start and end indices for slicing
    start_x = max(0, -x1)
    end_x = min(array.shape[0] - x1, x2 - x1)
    start_y = max(0, -y1)
    end_y = min(array.shape[1] - y1, y2 - y1)
    
    # Copy the overlapping area
    result[start_x:end_x, start_y:end_y] = array[max(x1, 0):max(x1, 0) + end_x - start_x, max(y1, 0):max(y1, 0) + end_y - start_y]
    return result


class Controller():
    """
    This class combines the Snake, Food, and Grid classes to handle the game logic.
    """

    def __init__(self, grid_size=[30,30], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, window_size = 11, maze_type = None, random_init=None):

        assert n_snakes < grid_size[0]//3
        assert n_snakes < 25
        assert snake_size < grid_size[1]//2
        assert unit_gap >= 0 and unit_gap < unit_size

        self.snakes_remaining = n_snakes
        self.grid = Grid(grid_size, unit_size, unit_gap)

        if maze_type is not None:
            if maze_type == 0:
                self.grid.grid[10:30, 10:30] = self.grid.BODY_COLOR
            
            if maze_type == 1:
                self.grid.grid[5:6, 5:15] = self.grid.BODY_COLOR
                self.grid.grid[5:6, 25:35] = self.grid.BODY_COLOR
                self.grid.grid[5:35, 35:36] = self.grid.BODY_COLOR
                self.grid.grid[5:35, 5:6] = self.grid.BODY_COLOR
                self.grid.grid[35:36, 5:36] = self.grid.BODY_COLOR
                self.grid.grid[5:20, 25:26] = self.grid.BODY_COLOR
                self.grid.grid[20:21, 10:26] = self.grid.BODY_COLOR

        self.snakes = []
        self.dead_snakes = []
        for i in range(1,n_snakes+1):
            hue = ((i+2) / (n_snakes + 2)) % 1.0
            rgb_color = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
            start_coord = [i*grid_size[0]//(n_snakes+1), snake_size+1]
            self.snakes.append(Snake(start_coord, snake_size))
            color = rgb_color
            self.snakes[-1].head_color = self.grid.HEAD_COLOR
            self.grid.draw_snake(self.snakes[-1], self.grid.HEAD_COLOR)
            self.dead_snakes.append(None)

        if random_init is not None:
            for i in range(2,n_foods+2):
                start_coord = [i*grid_size[0]//(n_foods+3), grid_size[1]-5]
                self.grid.place_food(start_coord)
        else:
            for i in range(n_foods):
                self.grid.new_food()

        self.window_size = window_size

        assert(self.window_size%2 == 1)

    def move_snake(self, direction, snake_idx):
        """
        Moves the specified snake according to the game's rules dependent on the direction.
        Does not draw head and does not check for reward scenarios. See move_result for these
        functionalities.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return

        # Cover old head position with body
        self.grid.cover(snake.head, self.grid.BODY_COLOR)
        # Erase tail without popping so as to redraw if food eaten
        self.grid.erase(snake.body[0])
        # Find and set next head position conditioned on direction
        snake.action(direction)

    def move_result(self, direction, snake_idx=0):
        """
        Checks for food and death collisions after moving snake. Draws head of snake if
        no death scenarios.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return 0

        # Check for death of snake
        if self.grid.check_death(snake.head):
            self.dead_snakes[snake_idx] = self.snakes[snake_idx]
            self.snakes[snake_idx] = None
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            self.grid.connect(snake.body.popleft(), snake.body[0], self.grid.SPACE_COLOR)
            reward = -1
        # Check for reward
        elif self.grid.food_space(snake.head):
            self.grid.draw(snake.body[0], self.grid.BODY_COLOR) # Redraw tail
            self.grid.connect(snake.body[0], snake.body[1], self.grid.BODY_COLOR)
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            reward = 1
            self.grid.new_food()
        else:
            reward = 0
            empty_coord = snake.body.popleft()
            self.grid.connect(empty_coord, snake.body[0], self.grid.SPACE_COLOR)
            self.grid.draw(snake.head, snake.head_color)

        self.grid.connect(snake.body[-1], snake.head, self.grid.BODY_COLOR)

        return reward
    
    def get_obs_single_snake(self, snake_idx):
        if self.snakes[snake_idx] is None:
            return np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
        
        return extract_and_pad(
            self.grid.grid.copy(),
            self.snakes[snake_idx].head[0] - self.window_size//2,
            self.snakes[snake_idx].head[0] + self.window_size//2 + 1,
            self.snakes[snake_idx].head[1] - self.window_size//2,
            self.snakes[snake_idx].head[1] + self.window_size//2 + 1,
        )
    
    def get_obs(self):
        return np.concatenate([self.get_obs_single_snake(i) for i in range(len(self.snakes))])

    def kill_snake(self, snake_idx):
        """
        Deletes snake from game and subtracts from the snake_count 
        """
        
        assert self.dead_snakes[snake_idx] is not None
        self.grid.erase(self.dead_snakes[snake_idx].head)
        self.grid.erase_snake_body(self.dead_snakes[snake_idx])
        self.dead_snakes[snake_idx] = None
        self.snakes_remaining -= 1

    def step(self, directions):
        """
        Takes an action for each snake in the specified direction and collects their rewards
        and dones.

        directions - tuple, list, or ndarray of directions corresponding to each snake.
        """

        # Ensure no more play until reset
        if self.snakes_remaining < 1 or self.grid.open_space < 1:
            if type(directions) == type(int()) or len(directions) is 1:
                return self.get_obs(), 0, True, {}
            else:
                return self.get_obs(), [0]*len(directions), True, {}

        rewards = []

        if isinstance(directions, np.int64) or isinstance(directions, int):
            directions = [directions]

        for i, direction in enumerate(directions):
            if self.snakes[i] is None and self.dead_snakes[i] is not None:
                self.kill_snake(i)
            self.move_snake(direction,i)
            rewards.append(self.move_result(direction, i))

        done = self.snakes_remaining < 1 or self.grid.open_space < 1
        if len(rewards) is 1:
            return self.get_obs(), rewards[0], done, {}
        else:
            return self.get_obs(), rewards, done, {}
