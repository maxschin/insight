import os

from torch.autograd import variable
from utils.hackatari_env import HackAtariWrapper
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import torch
import numpy as np
import json

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def save_equations(expression_list, output_folder, run_name):
    """
    Save a list of equations to a text file in the specified output folder.

    Parameters:
    - expression_list (list): List of string equations to save.
    - output_folder (str): Path to the folder where the file will be saved.
    - run_name (str): A name to include in the filename, e.g., identifying this run.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the output filename
    file_name = f"{run_name}_equations.txt"
    file_path = os.path.join(output_folder, file_name)
    
    # Write each equation to the file, one per line
    with open(file_path, 'w') as f:
        for equation in expression_list:
            f.write(equation + "\n\n")
    
    print(f"Equations saved to: {file_path}")


def get_reward_func_path(game, reward_func_name):
    """
    Build and validate the full file path for a reward function.
    
    The expected file structure is:
        reward_functions/<game>/<reward_func_name>.py
    
    Parameters:
        game (str): Name of the game directory under 'rewards'.
        reward_func_name (str): Base name of the reward function file (without .py).
    
    Returns:
        str: The absolute path to the reward function file.
    
    Raises:
        ValueError: If the game or reward_func_name contains invalid characters.
        FileNotFoundError: If the game directory or reward function file does not exist.
    """
    base_dir = os.path.join(SRC, "reward_functions")
    
    # Validate that game and reward_func_name don't contain path separators
    for part, name in [(game, "game"), (reward_func_name, "reward_func_name")]:
        if os.path.sep in part or "/" in part or "\\" in part:
            raise ValueError(f"Invalid {name!r}. It should not contain path separators.")

    # Construct the game directory path and validate it exists.
    game_path = os.path.join(base_dir, game)
    if not os.path.isdir(game_path):
        raise FileNotFoundError(f"Game directory does not exist: {game_path}")
    
    # Construct the full file path and validate the file exists.
    file_name = reward_func_name + ".py"
    file_path = os.path.join(game_path, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Reward function file does not exist: {file_path}")
    
    # Return the absolute path
    return os.path.abspath(file_path)


def make_env(env_name, seed,rewardfunc_path, modifs=[], sb3=False, pix=False, args=None):
    def thunk():
        env = HackAtariWrapper(
            env_name,
            modifs=modifs,
            rewardfunc_path=rewardfunc_path,
            frameskip=1,
            obs_mode="ori" if pix else "obj"
        )
        if not sb3:
            env = NoopResetEnv(env, noop_max=30) 
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if env.unwrapped.get_action_meanings()[1] == "FIRE":
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            if pix and args:
                env = gym.wrappers.ResizeObservation(env, (args.resolution, args.resolution))
                if args.gray:
                    env = gym.wrappers.GrayscaleObservation(env)
                env = gym.wrappers.FrameStackObservation(env, 4)
        env = Monitor(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def eval_policy(envs, action_func, device="cuda", n_episode=10):
    obs = envs.reset()
    total_return = 0.0
    total_length = 0.0
    total_episodes = 0

    while total_episodes<n_episode:
        obs_tensor = torch.Tensor(obs).to(device)
        with torch.no_grad():
            action = action_func(obs_tensor)
        obs, rewards, dones, infos = envs.step(action.cpu().numpy())

        done_indices = np.nonzero(dones)[0]
        if done_indices.size > 0:
            episode_returns = np.array([infos[i]["episode"]["r"] for i in done_indices])
            episode_lengths = np.array([infos[i]["episode"]["l"] for i in done_indices])
            total_return += episode_returns.sum()
            total_length += episode_lengths.sum()
            total_episodes += len(done_indices)
    avg_return = total_return / total_episodes
    avg_length = total_length / total_episodes

    return avg_return, avg_length

def get_object_order(game):
    folder = os.path.join(SRC, "batch_training", game)
    file = os.path.join(folder, "object_order.json")
    with open(file, 'r') as f:
        data = json.load(f)
    obj_order_raw = data["object_order"]
    num_obj_per_type = data["number_of_objects_per_type"]

    # construct obj names from obj order and num_obj_per_type
    objs_order = []
    for obj in obj_order_raw:
        for i in range(num_obj_per_type):
            if i == 0:
                objs_order.append(obj)
            else:
                objs_order.append(f"{obj}{str(i)}")

    # build variable names
    num_frames = 4
    num_objs = 256
    variable_names = []
    coords_order = ["y", "x"]
    for i in range(num_frames):
        frame = i+1
        for j in range(num_objs):
            for coord in coords_order:
                obj = f"obj{j+1}" if j>=len(objs_order) else objs_order[j]
                variable_name = f"{obj}_{coord}_{frame}"
                variable_names.append(variable_name)

    return variable_names



