
from dataclasses import dataclass
from gym_snake.envs.snake_env import SnakeEnv
from run_utility import load_model, show_progress_multi_agent
from testing import load_agents
import tyro


@dataclass
class Args:
    grid_dim: int = 40

    num_food: int = 30

    window_size: int = 11

    show_progress: bool = False

    num_snakes: int = 1


if __name__ == "__main__":
    args = tyro.cli(Args)

    agents = list(load_agents().values())

    print(agents)

    show_progress_multi_agent([agents[4],agents[4]], grid_dim = args.grid_dim, num_food = args.num_food, window_size = args.window_size, num_steps = 10000)