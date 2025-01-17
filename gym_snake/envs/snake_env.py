import os, subprocess, time, signal
import gym
from gymnasium import spaces as gymnasium_spaces
from gym import error, spaces, utils
from gym.utils import seeding
from gym_snake.envs.snake import Controller, Discrete

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: see matplotlib documentation for installation https://matplotlib.org/faq/installing_faq.html#installation".format(e))

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=[15,15], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, window_size = 11, maze_type = None, random_init=None):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.viewer = None
        self.action_space = gymnasium_spaces.Discrete(4)
        self.random_init = random_init

        self.maze_type = maze_type

        self.window_size = window_size

        self.observation_space = gymnasium_spaces.Box(low = 0, high = 255, shape = (self.window_size*self.unit_size, self.window_size*self.unit_size, 3))

    def step(self, action):
        self.last_obs, rewards, done, info= self.controller.step(action)
        return self.last_obs, rewards, done, False, {}

    def reset(self, seed = None, **kwargs):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, window_size = self.window_size, maze_type = self.maze_type, random_init=self.random_init)
        self.last_obs = self.controller.get_obs()
        return self.last_obs, {}

    def render(self, mode='human', close=False, frame_speed=0.1):
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()
        else:
            self.viewer.clear()
            self.viewer.imshow(self.controller.grid.grid)
            plt.pause(frame_speed)
        self.fig.canvas.draw()

    def seed(self, x):
        pass
