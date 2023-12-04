from gym_snake.envs.snake_env import SnakeEnv

import torch


def show_progress(agent, grid_dim = 10, num_food = 1, window_size = 11, num_steps = 100):
    env = SnakeEnv(grid_size = [grid_dim, grid_dim], unit_size = 1, n_foods = num_food,  unit_gap = 0, n_snakes = 1, snake_size = 3, window_size = window_size)
    next_obs, _ = env.reset()
    print(next_obs)

    for i in range(num_steps):
        env.render(frame_speed = 0.001)
        action, _, _, _ = agent.get_action_and_value(torch.tensor(next_obs).unsqueeze(0))
        next_obs, reward, terminations, truncations, infos = env.step(action.cpu().numpy())

        if terminations or truncations:
            if num_steps > 100000:
                return
            
            next_obs, _ = env.reset()


def get_snake_obs(next_obs, snake_idx, window_size):
    return torch.tensor(next_obs[window_size * snake_idx : window_size * (snake_idx + 1)]).unsqueeze(0)


def show_progress_multi_agent(agents, grid_dim = 10, num_food = 1, window_size = 11, num_steps = 100):
    env = SnakeEnv(
        grid_size = [grid_dim, grid_dim], 
        unit_size = 1, 
        n_foods = num_food,  
        unit_gap = 0, 
        n_snakes = len(agents), 
        snake_size = 3, 
        window_size = window_size,
    )

    next_obs, _ = env.reset()
    print(next_obs)

    for i in range(num_steps):
        env.render(frame_speed = 0.001)
        
        actions = []
        for snake_idx, agent in enumerate(agents):
            action, _, _, _ = agent.get_action_and_value(get_snake_obs(next_obs, snake_idx, window_size))
            actions.append(action.cpu().numpy())

        next_obs, reward, terminations, truncations, infos = env.step(actions)

        if terminations or truncations:
            if num_steps > 100000:
                return
            
            next_obs, _ = env.reset()


def save_model(agent, path):
    torch.save(agent.state_dict(), path)

def load_model(agent, path):
    agent.load_state_dict(torch.load(path))