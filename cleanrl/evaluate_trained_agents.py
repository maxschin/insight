import argparse
import os
import torch
import pandas as pd
from stable_baselines3.common.vec_env import SubprocVecEnv
from hackatari_env import HackAtariWrapper  # Ensure this is correctly imported

MODIFS = {
  "Pong": ["default", "lazy_enemy"],
  "Freeway": [],
  "Seaquest": []
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Atari agents from checkpoints.")
    parser.add_argument("--run_names", nargs="*", type=str, help="List of run names (filenames without extension).")
    parser.add_argument("--record_eql", action="store_true", help="Flag to record equality check logs.")
    parser.add_argument("--use_modifs", action="store_true", help="Flag to enable modifications.")
    return parser.parse_args()

def make_env(env_name, seed, rewardfunc_path, modifs, index, is_eval=False, video_folder=None):
    def thunk():
        env = HackAtariWrapper(env_name, modifs=modifs, rewardfunc_path=rewardfunc_path)
        env.reset(seed=seed)
        return env
    return thunk

def load_checkpoints(run_names):
    checkpoint_list = []
    ckpt_dir = os.path.abspath("models/agents")
    
    if run_names:
        filenames = [f"{name}_final.pth" for name in run_names]
    else:
        filenames = [f for f in os.listdir(ckpt_dir) if f.endswith("_final.pth")]
        run_names = [f.rsplit("_final.pth", 1)[0] for f in filenames if "_final.pth" in f]
    
    for filename in filenames:
        path = os.path.join(ckpt_dir, filename)
        if os.path.exists(path):
            agent = torch.load(path)
            run_name = filename.rsplit("_final.pth", 1)[0]
            parts = run_name.split("_")
            game = parts[0] if len(parts) > 0 else None
            agent_type = parts[1] if len(parts) > 1 else None
            reward_function = parts[2] if len(parts) > 2 else "default"
            
            checkpoint_list.append({
                "run_name": run_name,
                "agent": agent,
                "agent_type": agent_type,
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
    checkpoints = load_checkpoints(args.run_names)
    
    for checkpoint in checkpoints:
        name = checkpoint["run_name"]
        agent = checkpoint.pop("agent")  # Remove agent to avoid storing it in the DataFrame
        game = checkpoint["game"]
        
        modifications = MODIFS.get(game, ["default"]) if args.use_modifs else ["default"]
        for modif in modifications:
            env = SubprocVecEnv([make_env(game, seed=42, rewardfunc_path=checkpoint["reward_function"], modifs=[modif] if modif != "default" else [], index=i) for i in range(4)])
            
            modif_key = modif if modif != "default" else ""
            checkpoint[f"{modif_key}_neural" if modif_key else "neural"] = evaluate_agent(agent, env, episodes=10, actor="neural")
            checkpoint[f"{modif_key}_eql" if modif_key else "eql"] = evaluate_agent(agent, env, episodes=10, actor="eql")
            
            if args.record_eql:
                base_folder = 'ppoeql_ocatari_videos'
                os.makedirs(base_folder, exist_ok=True)
                run_folder = os.path.join(base_folder, name, modif if modif != "default" else "baseline")
                os.makedirs(run_folder, exist_ok=True)
                record_folder = os.path.join(run_folder, 'eval')
                os.makedirs(record_folder, exist_ok=True)
                
                label = f"{checkpoint['reward_function']}_{modif}" if modif != "default" else checkpoint['reward_function']
                visual_for_ocatari_agent_videos(env, agent, "cpu", args, record_folder, n_step=1000, actor="eql", label=label)
    
    df = pd.DataFrame(checkpoints)
    df.to_csv("evaluation_results.csv", index=False)
    print("Saved evaluation results to evaluation_results.csv")

if __name__ == "__main__":
    main()
