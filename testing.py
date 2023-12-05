
import numpy as np
from agent import Agent1, Agent2, Agent3
from gym_snake.envs.snake_env import SnakeEnv
from run_utility import load_model, show_progress_multi_agent
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

NUM_SAMPLES = 100


def get_snake_obs(next_obs, snake_idx, window_size):
    return torch.tensor(next_obs[window_size * snake_idx : window_size * (snake_idx + 1)]).unsqueeze(0)

def show_progress_multi_agent(agents, env, window_size = 11):
    next_obs, _ = env.reset()

    while True:
        env.render(frame_speed = 0.001)
        
        actions = []
        for snake_idx, agent in enumerate(agents):
            action, _, _, _ = agent.get_action_and_value(get_snake_obs(next_obs, snake_idx, window_size))
            actions.append(action.cpu().numpy())

        next_obs, reward, terminations, truncations, infos = env.step(actions)

        if terminations or truncations:
            return

def multi_agent_death_length(agents, env, window_size = 11):
    next_obs, _ = env.reset()

    if len(agents) == 1:
        lengths = 3
    else:
        lengths = [3] * len(agents)

    while True:
        actions = []
        for snake_idx, agent in enumerate(agents):
            action, _, _, _ = agent.get_action_and_value(get_snake_obs(next_obs, snake_idx, window_size))
            actions.append(action.cpu().numpy())

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        if len(agents) == 1:
            lengths += rewards
        else:
            lengths = [length + reward for length, reward in zip(lengths, rewards)]

        if terminations or truncations:
            if len(agents) == 1:
                return lengths + 1
            else:
                return [length + 1 for length in lengths]
            
def multi_agent_stats(agents, env, num_samples = 100, window_size = 11):
    stats = []
    for i in tqdm(range(num_samples)):
        stats.append(multi_agent_death_length(agents, env, window_size))


    print(f"Snake length: {np.mean(stats)}±{np.std(stats)}")

    plt.hist(stats, bins = 20)
    plt.show()

    return stats

def test(agent, env):
    show_progress_multi_agent(
        [agent],  
        env,
        window_size = 11,
    )

    lengths = multi_agent_stats(
        [agent], 
        env, 
        window_size = 11,
        num_samples = NUM_SAMPLES,
    )

def test_basic_medium_density(agent):
    env = SnakeEnv(
        grid_size = [40, 40], 
        unit_size = 1, 
        n_foods = 30,  
        unit_gap = 0, 
        n_snakes = 1, 
        snake_size = 3, 
        window_size = 11,
    )

    test(agent, env)

def test_basic_low_density(agent):
    env = SnakeEnv(
        grid_size = [40, 40], 
        unit_size = 1, 
        n_foods = 10,  
        unit_gap = 0, 
        n_snakes = 1, 
        snake_size = 3, 
        window_size = 11,
    )

    test(agent, env)

def test_basic_high_density(agent):
    env = SnakeEnv(
        grid_size = [40, 40], 
        unit_size = 1, 
        n_foods = 100,  
        unit_gap = 0, 
        n_snakes = 1, 
        snake_size = 3, 
        window_size = 11,
    )

    test(agent, env)

def test_maze0_medium_density(agent):
    env = SnakeEnv(
        grid_size = [40, 40], 
        unit_size = 1, 
        n_foods = 30,  
        unit_gap = 0, 
        n_snakes = 1, 
        snake_size = 3, 
        maze_type=0,
        window_size = 11,
    )

    test(agent, env)

def test_maze1_medium_density(agent):
    env = SnakeEnv(
        grid_size = [40, 40], 
        unit_size = 1, 
        n_foods = 30,  
        unit_gap = 0, 
        n_snakes = 1, 
        snake_size = 3, 
        maze_type=1,
        window_size = 11,
    )

    env.reset()

    plt.imshow(env.controller.grid.grid)
    plt.show()

    test(agent, env)


agent = Agent3(4)
load_model(agent, 'models/Agent_3_Length_10.pt')
test_maze1_medium_density(agent)