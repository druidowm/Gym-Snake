
from dataclasses import dataclass
from agent import Agent
from gym_snake.envs.snake_env import SnakeEnv
from run_utility import load_model, show_progress_multi_agent
import tyro


@dataclass
class Args:
    grid_dim: int = 10

    num_food: int = 1

    window_size: int = 11

    show_progress: bool = False

    num_snakes: int = 1


if __name__ == "__main__":
    args = tyro.cli(Args)

    agent = Agent(4)
    load_model(agent, 'models/good_run__1__window_size=11__1701730527.pt')

    show_progress_multi_agent([agent]*args.num_snakes, grid_dim = args.grid_dim, num_food = args.num_food, window_size = args.window_size, num_steps = 10000)