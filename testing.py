
from dataclasses import dataclass
import pickle
import numpy as np
from agent import Agent1, Agent2, Agent3, Agent4, Agent5
from gym_snake.envs.snake_env import SnakeEnv
from run_utility import load_model, show_progress_multi_agent
import torch
from tqdm import tqdm
import tyro

import matplotlib.pyplot as plt

NUM_SAMPLES = 100


@dataclass
class Args:
    multi: bool = False

def get_snake_obs(next_obs, snake_idx, window_size):
    return torch.tensor(next_obs[window_size * snake_idx : window_size * (snake_idx + 1)]).unsqueeze(0)

def show_progress_multi_agent(agents, env, window_size = 11):
    next_obs, _ = env.reset()

    try:
        while True:
            env.render(frame_speed = 0.001)
            
            actions = []
            for snake_idx, agent in enumerate(agents):
                #print(get_snake_obs(next_obs, snake_idx, window_size))
                action, _, _, _ = agent.get_action_and_value(get_snake_obs(next_obs, snake_idx, window_size))
                actions.append(action.cpu().numpy())

            next_obs, reward, terminations, truncations, infos = env.step(actions)

            if terminations or truncations:
                return
    except:
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

    #plt.hist(stats, bins = 20)
    #plt.show()

    return stats

def test(agent, env):
    # show_progress_multi_agent(
    #     [agent] if not isinstance(agent, list) else agent,
    #     env,
    #     window_size = 11,
    # )

    return multi_agent_stats(
        [agent] if not isinstance(agent, list) else agent,
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
        n_snakes = len(agent) if isinstance(agent, list) else 1,
        snake_size = 3, 
        window_size = 11,
    )

    return test(agent, env)

def test_basic_low_density(agent):
    env = SnakeEnv(
        grid_size = [40, 40], 
        unit_size = 1, 
        n_foods = 10,  
        unit_gap = 0, 
        n_snakes = len(agent) if isinstance(agent, list) else 1,
        snake_size = 3, 
        window_size = 11,
    )

    return test(agent, env)

def test_basic_high_density(agent):
    env = SnakeEnv(
        grid_size = [40, 40], 
        unit_size = 1, 
        n_foods = 100,  
        unit_gap = 0, 
        n_snakes = len(agent) if isinstance(agent, list) else 1,
        snake_size = 3, 
        window_size = 11,
    )

    return test(agent, env)

def test_maze0_medium_density(agent):
    env = SnakeEnv(
        grid_size = [40, 40], 
        unit_size = 1, 
        n_foods = 30,  
        unit_gap = 0, 
        n_snakes = len(agent) if isinstance(agent, list) else 1,
        snake_size = 3, 
        maze_type=0,
        window_size = 11,
    )

    return test(agent, env)

def test_maze1_medium_density(agent):
    env = SnakeEnv(
        grid_size = [40, 40], 
        unit_size = 1, 
        n_foods = 30,  
        unit_gap = 0, 
        n_snakes = len(agent) if isinstance(agent, list) else 1,
        snake_size = 3, 
        maze_type=1,
        window_size = 11,
    )

    return test(agent, env)

def battle_royale(agents):
    env = SnakeEnv(
        grid_size = [60, 60], 
        unit_size = 1, 
        n_foods = 100,  
        unit_gap = 0, 
        n_snakes = len(agents), 
        snake_size = 3, 
        window_size = 11,
    )

    test(agents, env)

def load_agents():
    agent1_3 = Agent1(4)
    load_model(agent1_3, 'models/Agent_1_Length_3.pt')
    agent2_3 = Agent2(4)
    load_model(agent2_3, 'models/Agent_2_Length_3.pt')
    agent3_3 = Agent3(4)
    load_model(agent3_3, 'models/Agent_3_Length_3.pt')
    agent4_3 = Agent4(4)
    load_model(agent4_3, 'models/Agent_4_Length_3.pt')
    agent5_3 = Agent5(4)
    load_model(agent5_3, 'models/Agent_5_Length_3.pt')

    agent1_10 = Agent1(4)
    load_model(agent1_10, 'models/Agent_1_Length_10.pt')
    agent2_10 = Agent2(4)
    load_model(agent2_10, 'models/Agent_2_Length_10.pt')
    agent3_10 = Agent3(4)
    load_model(agent3_10, 'models/Agent_3_Length_10.pt')
    agent4_10 = Agent4(4)
    load_model(agent4_10, 'models/Agent_4_Length_10.pt')
    agent5_10 = Agent5(4)
    load_model(agent5_10, 'models/Agent_5_Length_10.pt')

    return {
        "agent1_3" : agent1_3,
        "agent2_3" : agent2_3,
        "agent3_3" : agent3_3,
        "agent4_3" : agent4_3,
        "agent5_3" : agent5_3,
        "agent1_10" : agent1_10,
        "agent2_10" : agent2_10,
        "agent3_10" : agent3_10,
        "agent4_10" : agent4_10,
        "agent5_10" : agent5_10,
    }


def main(multi):
    agents = load_agents()

    if not multi:

        tests = [
            test_basic_medium_density,
            #test_basic_low_density,
            #test_basic_high_density,
            test_maze0_medium_density,
            test_maze1_medium_density,
        ]

        results = {}

        for test in tests:
            results[test.__name__] = {
                "agent1_3" : [],
                "agent2_3" : [],
                "agent3_3" : [],
                "agent4_3" : [],
                "agent5_3" : [],
                # "agent1_10" : [],
                "agent2_10" : [],
                # "agent3_10" : [],
                # "agent4_10" : [],
                # "agent5_10" : [],
            }

        for test in tests:
            print(f"Running test {test.__name__}")
            
            for key in agents:
                print(f"Agent {key}")
                results[test.__name__][key] = test(agents[key])
                pickle.dump(results, open("results.pkl", "wb"))

        pickle.dump(results, open("results_single.pkl", "wb"))

    else:

        tests = [
            test_basic_medium_density,
            #test_basic_low_density,
            #test_basic_high_density,
            #test_maze0_medium_density,
            #test_maze1_medium_density,
        ]

        results = {}

        for test in tests:
            results[test.__name__] = {
                "agent1_3" : [],
                "agent2_3" : [],
                "agent3_3" : [],
                "agent4_3" : [],
                "agent5_3" : [],
                # "agent1_10" : [],
                "agent2_10" : [],
                # "agent3_10" : [],
                # "agent4_10" : [],
                # "agent5_10" : [],
            }

        for test in tests:
            print(f"Running test {test.__name__}")
            
            for key in agents:
                print(f"Agent {key}")
                results[test.__name__][key] = test([agents[key]]*5)
                pickle.dump(results, open("results_multi.pkl", "wb"))

                print(results[test.__name__][key])

        pickle.dump(results, open("results_multi.pkl", "wb"))

if __name__ == "__main__":
    args = tyro.cli(Args)
    #main(args.multi)

    agents = list(load_agents().values())[:5]

    print(agents)

    pickle.dump(battle_royale(agents), open("battle.pkl", "wb"))