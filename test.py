from gym_snake.envs.snake_env import SnakeEnv

env = SnakeEnv(grid_size = [100,100], n_snakes = 10, snake_size = 30)
env.reset()

for i in range(100):
    env.render()
    env.step([env.action_space.sample() for i in range(env.n_snakes)]) # take a random action