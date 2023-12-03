from gym_snake.envs.snake_env import SnakeEnv

env = SnakeEnv(grid_size = [20,20], n_snakes = 4, snake_size = 3, unit_gap=0, unit_size=1)
env.reset()

for i in range(100):
    env.render()
    env.step([env.action_space.sample() for i in range(env.n_snakes)]) # take a random action