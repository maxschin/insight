import os
import sys
SRC = os.path.abspath(os.path.dirname(__file__))
sys.path.append(SRC)

import re
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import pandas as pd
from utils.utils import get_reward_func_path, eval_policy, make_env
import copy
from utils.visualize_utils import visual_for_ocatari_agent_videos
from distutils.util import strtobool
from tqdm import tqdm
from rtpt import RTPT

MODIFS = {
  "PongNoFrameskip-v4": ["default", "lazy_enemy"],
  "SeaquestNoFrameskip-v4": ["default", "disable_enemies"],
  "MsPacmanNoFrameskip-v4": ["default", "set_level_1","set_level_2","set_level_3"],
  "SpaceInvadersNoFrameskip-v4": ["default", "shift_shields_three"],
  "FreewayNoFrameskip-v4": ["default", "stop_all_cars"],
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Atari agents from checkpoints.")
    parser.add_argument("--run_names", nargs="*", type=str, help="List of run names (filenames without extension).")
    parser.add_argument("--record_eql", type=lambda x: bool(strtobool(x)), default=True, help="Flag to record equality check logs.")
    parser.add_argument("--use_modifs", type=lambda x: bool(strtobool(x)), default=True, help="Flag to enable Hackatari modifications")
    parser.add_argument("--use_e2e", type=lambda x: bool(strtobool(x)), default=True, help="Evalute end-to-end agents, otherwise object-enctric agents")
    parser.add_argument("--n_envs_eval", type=int, default=4, help="Number of parallel envs to use for evaluation")
    parser.add_argument("--n_eval_episodes", type=int, default=30, help="Number of episodes that returns should be collected on")
    parser.add_argument("--n_record_steps", type=int, default=1000, help="Number of steps that should be taken per recording in environment")
    parser.add_argument("--seed", type=int, default=21, help="The seed to use for the eval envs")
    # for debugging
    parser.add_argument("--local_debugging", action="store_true", help="Radically shortens the training loop when running outside of container")
    return parser.parse_args()

def load_checkpoints(run_names, device="cpu", use_e2e=True):
    checkpoint_list = []
    ckpt_dir = os.path.join(SRC, "models", "agents")
    
    ending = f'_{"e2e" if use_e2e else "oc"}_final.pth'
    if run_names:
        filenames = [f"{name}{ending}" for name in run_names]
    else:
        filenames = [f for f in os.listdir(ckpt_dir) if f.endswith(ending)]
        run_names = [f.rsplit(ending, 1)[0] for f in filenames if ending in f]
        print(run_names)
    
    for filename in filenames:
        path = os.path.join(ckpt_dir, filename)
        if os.path.exists(path):
            agent = torch.load(path, weights_only=False, map_location=device)
            pattern = re.compile(
                r'^'
                r'(?P<game>.+?)'                # 1. game name, non‑greedy
                r'_Agent'
                r'(?:_(?P<reward_function>'     # 2. optional reward_function
                  r'(?!e2e|oc).+?'              #    — but not “e2e” or “oc”
                r'))?'
                r'(?:_(?:e2e|oc))?'             # 3. optional suffix “_e2e” or “_oc”
                r'_final\.pth'
                r'$'
            )
            match = re.match(pattern, filename)
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

def main():
    args = parse_args()

    # check if running inside container
    containerized = os.environ.get("container") == "podman"
    if containerized:
        print("Running inside container")
    else:
        if args.local_debugging:
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
    args.n_layers = 1
    
    # collect data
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
                envs = SubprocVecEnv([make_env(args.game, args.seed + i, modifs=modif_list, rewardfunc_path=rewardfunc_path, pix=True, args=args) for i in range(args.n_envs_eval)], start_method="fork")

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
