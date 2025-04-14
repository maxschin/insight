import re
import argparse
import os
from stable_baselines3.common.monitor import Monitor
import torch
import pandas as pd
from hackatari_env import SyncVectorEnvWrapper  # Ensure this is correctly imported
from hackatari_utils import get_reward_func_path
from train_policy_atari import make_env, eval_policy
import copy
from visualize_utils import visual_for_ocatari_agent_videos
from distutils.util import strtobool
from tqdm import tqdm
from rtpt import RTPT

MODIFS = {
  "PongNoFrameskip-v4": ["default", "lazy_enemy", "hidden_enemy", "up_drift", "left_drift"],
  "Freeway": [],
  "Seaquest": []
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Atari agents from checkpoints.")
    parser.add_argument("--run_names", nargs="*", type=str, help="List of run names (filenames without extension).")
    parser.add_argument("--record_eql", type=lambda x: bool(strtobool(x)), default=True, help="Flag to record equality check logs.")
    parser.add_argument("--use_modifs", type=lambda x: bool(strtobool(x)), default=True, help="Flag to enable Hackatari modifications")
    parser.add_argument("--n_envs_eval", type=int, default=4, help="Number of parallel envs to use for evaluation")
    parser.add_argument("--n_eval_episodes", type=int, default=30, help="Number of episodes that returns should be collected on")
    parser.add_argument("--n_record_steps", type=int, default=1000, help="Number of steps that should be taken per recording in environment")
    parser.add_argument("--seed", type=int, default=21, help="The seed to use for the eval envs")
    return parser.parse_args()

def load_checkpoints(run_names, device="cpu"):
    checkpoint_list = []
    ckpt_dir = os.path.abspath("models/agents")
    
    if run_names:
        filenames = [f"{name}_final.pth" for name in run_names]
    else:
        filenames = [f for f in os.listdir(ckpt_dir) if f.endswith("_final.pth")]
        run_names = [f.rsplit("_final.pth", 1)[0] for f in filenames if "_final.pth" in f]
        print(run_names)
    
    for filename in filenames:
        path = os.path.join(ckpt_dir, filename)
        if os.path.exists(path):
            agent = torch.load(path, weights_only=False, map_location=device)
            match = re.match(r"^(?P<game>.*?)_Agent(?:_(?P<reward_function>.*?))?_final\.pth$", filename)
            game = match.group("game")
            reward_function = match.group("reward_function")
            reward_function = reward_function if reward_function else "default"
            run_name = f"{game}_Agent_{reward_function}" if reward_function else game

            print("Game:", game)
            print("Reward function:", reward_function or None)
            print("Run name:", run_name)
            checkpoint_list.append({
                "run_name": run_name,
                "agent": agent,
                "agent_type": "Agent",
                "game": game,
                "reward_function": reward_function
            })
    
    return checkpoint_list

def evaluate_agent(agent, env, episodes=10, actor="neural", device="cpu"):
    """
    Evaluate an agent using a vectorized environment (SubprocVecEnv) over a total of `episodes` complete episodes.
    Handles cases where individual sub-environments finish at different times.
    
    :param agent: The agent to evaluate. Must have a method `get_action_and_value` that accepts a batch of observations.
    :param env: A vectorized environment (e.g., SubprocVecEnv) that returns batches of observations.
    :param episodes: Total number of episodes to average over.
    :param actor: Which actor network to use when getting actions.
    :return: Average episode reward.
    """
    if not isinstance(device, torch.device):
        device = "cpu"
    total_rewards = []  # Stores completed episode rewards
    # Reset all sub-environments
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    # Convert observations to torch tensor
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    # Initialize a reward accumulator for each sub-environment
    num_envs = obs.shape[0]
    ep_rewards = [0.0] * num_envs

    while len(total_rewards) < episodes:
        # Get actions for the batch of observations from the specified actor
        action, _, _, _, _, _ = agent.get_action_and_value(obs, actor=actor)
        # Convert actions to numpy as required by the vectorized env
        action_np = action.cpu().numpy()
        # Step the vectorized environment. These will be arrays/lists with length num_envs.
        next_obs, rewards, dones, infos = env.step(action_np)
        # If next_obs is a tuple, extract the observations.
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        # Loop through each sub-environment's results
        for i, done in enumerate(dones):
            ep_rewards[i] += rewards[i]
            if done:
                total_rewards.append(ep_rewards[i])
                ep_rewards[i] = 0.0  # Reset the accumulator for that environment

        # Prepare for next iteration: convert observations to torch tensor.
        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    # Compute the average reward over the collected episodes
    avg_reward = sum(total_rewards[:episodes]) / episodes
    return avg_reward

def main():
    args = parse_args()

    # check if running inside container
    containerized = os.environ.get("container") == "podman"
    if containerized:
        print("Running inside container")
    else:
        args.n_envs_eval = 4
        args.n_eval_episodes = 1
        args.use_modifs = False
        print("Running locally")

    # get device and then load agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = load_checkpoints(args.run_names, device=device)

    # insert args for make_env etc
    args.resolution = 84
    args.threshold = 0.5
    args.gray = True
    
    final_datapoints = []
    
    # iterate over all final checkpoints of agents to evaluate
    rtpt = RTPT(name_initials="MS", experiment_name="INSIGHT_EVAL", max_iterations=len(checkpoints))
    rtpt.start()
    with tqdm(total=len(checkpoints), desc="Evaluating trained agents") as pbar:
        for checkpoint in checkpoints:
            # extract meta-info
            name = checkpoint["run_name"]
            print(f"RUN NAME: {name}")
            agent = checkpoint.pop("agent")  # Remove agent to avoid storing it in the DataFrame
            game = checkpoint["game"]
            args.game = game
            reward_function = checkpoint["reward_function"]

            # load reward function to evaluate episode rewards
            rewardfunc_path = get_reward_func_path(game, reward_function) if reward_function != "default" else None
            
            # evaluate per modification available
            modifications = MODIFS.get(game, ["default"]) if args.use_modifs else ["default"]
            for modif in modifications:
                print(f"MOD: {modif}")
                # one modification, one line in the df...
                data_point = copy.deepcopy(checkpoint)
                data_point["modif"] = modif
                modif_list = [modif] if modif != "default" else []

                # set up envs for evaluation and recording
                envs = SyncVectorEnvWrapper(
                    [make_env(args.game, args.seed + i, args, rewardfunc_path, modifs=modif_list) for i in range(args.n_envs_eval)])

                # always record statistics for both types of agents
                for agent_type in ["neural", "eql"]:
                    print(agent_type)
                    action_func = lambda t: agent.get_action_and_value(t, threshold=args.threshold, actor=agent_type)[0]
                    mean_episode_reward, mean_episode_length = eval_policy(envs, action_func, device=device, n_episode=args.n_eval_episodes)
                    data_point[f"{agent_type}_returns"] = mean_episode_reward
                    data_point[f"{agent_type}_length"] = mean_episode_length

                # store differences between neural and eql agents
                data_point["eql_neural_reward_diff"] = data_point["eql_returns"] - data_point["neural_returns"]
                data_point["eql_neural_length_diff"] = data_point["eql_length"] - data_point["neural_length"]

                print(data_point)
                final_datapoints.append(data_point)
                
                if args.record_eql:
                    print("Starting recording")
                    base_folder = 'ppoeql_ocatari_videos'
                    os.makedirs(base_folder, exist_ok=True)
                    run_folder = os.path.join(base_folder, name)
                    os.makedirs(run_folder, exist_ok=True)
                    record_folder = os.path.join(run_folder, 'eval')
                    os.makedirs(record_folder, exist_ok=True)
                    
                    label = f"{checkpoint['reward_function']}_{modif}" if modif != "default" else checkpoint['reward_function']
                    visual_for_ocatari_agent_videos(envs, agent, device, args, record_folder, n_step=args.n_record_steps, actor="eql", label=label)
        pbar.update(1)
        rtpt.step()
    
    df = pd.DataFrame(final_datapoints)
    df.to_csv("evaluation_results.csv", index=False)
    print("Saved evaluation results to evaluation_results.csv")

if __name__ == "__main__":
    main()
