from gym_snake.envs.snake_env import SnakeEnv


def show_progress(grid_dim = 10, num_food = 1, window_size = 11):
    env = SnakeEnv(grid_size = [grid_dim, grid_dim], unit_size = 1, n_foods = num_food,  unit_gap = 0, n_snakes = 1, snake_size = 3, window_size = window_size)
    next_obs, _ = env.reset()
    print(next_obs)

    for i in range(100):
        print(env.controller.snakes[0].head)

        env.render(frame_speed = 0.001)
        input()

        action = env.action_space.sample()
        next_obs, reward, terminations, truncations, infos = env.step(action)

        if terminations or truncations:
            next_obs, _ = env.reset()


show_progress()