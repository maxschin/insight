# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from agents.eql.regularization import L12Smooth
from agents.agent import load_agent, AgentSimplified
from hackatari_env import HackAtariWrapper, SyncVectorEnvWrapper
from hackatari_utils import save_equations, get_reward_func_path

from torch.nn import CrossEntropyLoss

from visualize_utils import visual_for_ocatari_agent_videos
from tqdm import tqdm

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="insight",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Hackatari specific arguments
    parser.add_argument("-g", "--game", type=str,
                        default="Pong", help="Game to be run")
    parser.add_argument("-m", "--modifs", nargs="+", default=[],
                        help="List of modifications to the game")
    parser.add_argument("-rf", "--reward_function", type=str,
                        default="", help="Custom reward function file name")

    # Algorithm specific arguments
    parser.add_argument("--agent_type", type=str, default="AgentSimplified", help="class name of the agent to be used")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--run-name", type=str, default=None,
        help="the defined run_name")
    parser.add_argument("--reg_weight", type=float, default= 1e-3,
        help="regulization for interpertable")
    parser.add_argument("--use_nn", type=lambda x: bool(strtobool(x)), default=True,
        help="use nn for critic")
    parser.add_argument("--deter_action", type=lambda x: bool(strtobool(x)), default=False,
        help="deterministic action or not")
    parser.add_argument("--pre_nn_agent", type=lambda x: bool(strtobool(x)), default=False,
        help="load nn agent or not")
    parser.add_argument("--fix_cri", type=lambda x: bool(strtobool(x)), default=False,
        help="fix cri or not")
    parser.add_argument("--n_funcs", type=int, default=4,
        help="n_funcs")
    parser.add_argument("--n_layers", type=int, default=1,
        help="n_layers")
    parser.add_argument("--ng", type=lambda x: bool(strtobool(x)), default=True,
        help="neural guided or not")
    parser.add_argument("--visual", type=lambda x: bool(strtobool(x)), default=False,
        help="visualize or not")
    parser.add_argument("--save", type=lambda x: bool(strtobool(x)), default=True,
        help="save")
    parser.add_argument("--pre_train", type=lambda x: bool(strtobool(x)), default=False,
        help="pretrain agent or not")
    parser.add_argument("--pre_train_uptates", type=int, default=500,
        help="number of pre-train update")
    parser.add_argument("--clip_drop", type=lambda x: bool(strtobool(x)), default=False,
        help="drop clip-coef or not")
    parser.add_argument("--pnn_guide", type=lambda x: bool(strtobool(x)), default=False,
        help="use pure nn guide or not")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--distillation_loss_weight", type=float, default=1)
    parser.add_argument("--reg_weight_drop", type=lambda x: bool(strtobool(x)), default=True,
        help="drop reg weight or not")   
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def make_env(env_name, seed,rewardfunc_path, modifs):
    def thunk():
        env = HackAtariWrapper(env_name, modifs=modifs, rewardfunc_path=rewardfunc_path)  
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30) 
        # env = MaxAndSkipEnv(env, skip=4) -> done by hackatari by default
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def eval_policy(envs, action_func, device="cuda", n_episode=10):
    obs,_ = envs.reset()
    episode_return = 0
    episode_length = 0
    total_episode = 0
    while total_episode<n_episode:
        obs_tensor = torch.Tensor(obs).to(device)
        with torch.no_grad():
            action = action_func(obs_tensor)
            obs, _, _, _, info = envs.step(action.cpu().numpy())
        if "final_info" in info:
            for env_id, env_info in enumerate(info["final_info"]):
                if not env_info is None:
                    if "episode" in env_info:
                        episode_return += env_info["episode"]["r"]
                        episode_length += env_info["episode"]["l"]
                        total_episode += 1.
                        if total_episode == n_episode:
                            break
    return episode_return / total_episode, episode_length / total_episode


if __name__ == "__main__":
    execution_time = 0

    args = parse_args()
    if args.run_name == None:
        run_name = f"{args.game}" + f"_{args.agent_type}" 
        if args.reward_function:
            run_name += f"_{args.reward_function}"
        if args.modifs:
            run_name += f"_{''.join(args.modifs)}"
        #if args.pre_nn_agent:
        #    run_name+="_pre_nn_agent"
        #if args.ng:
        #    run_name+="_ng"
        #if args.pnn_guide:
        #    run_name+="_png"
        #run_name+=f"_seed{args.seed}"
    else:
        run_name = args.run_name
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    
    # Setup multiple OCAtari environments
    rewardfunc_path = get_reward_func_path(args.game, args.reward_function) if args.reward_function else None
    envs = SyncVectorEnvWrapper(
        [make_env(args.game, args.seed + i, modifs=args.modifs, rewardfunc_path=rewardfunc_path) for i in range(args.num_envs)])
    envs_eval = SyncVectorEnvWrapper(
        [make_env(args.game, args.seed + i, modifs=args.modifs, rewardfunc_path=rewardfunc_path) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    agent_in_dim = envs.get_ns_out_dim()


    agent_class = load_agent(args.agent_type)
    if args.pre_nn_agent:
        if args.pnn_guide:
            print('pnn_guide')
            agent_nn = torch.load('models/agents/'+f"{args.env_id}"+f'_{args.obj_vec_length}'+"_NN"+"_gray"*args.gray+f"_objs{args.n_objects}"+f"_seed{args.seed}"+'.pth').to(device)
        else:
            agent_nn = torch.load('models/agents/'+f"{args.env_id}"+f'_{args.obj_vec_length}'+f"_gray{args.gray}"+f"_t{args.total_timesteps}"+f"_objs{args.n_objects}"+f"_seed{args.seed}"+'.pth').to(device)
            print('nn_guide')
        agent = agent_class(envs,args,agent_nn, agent_in_dim=agent_in_dim, skip_perception=True).to(device)
        for param in agent_nn.parameters():
            param.requires_grad = False
    else:
        agent = agent_class(envs,args, agent_in_dim=agent_in_dim, skip_perception=True).to(device)

    #fix hypara
    if args.fix_cri:
            for param in agent.critic.parameters():
                param.requires_grad = False

    optimizer = optim.Adam(
        [{'params':agent.neural_actor.parameters(),'lr':args.learning_rate },
         {'params':agent.eql_actor.parameters(),'lr':args.learning_rate},
         {'params':agent.critic.parameters(),'lr':args.learning_rate},
         ], eps=1e-5)

    regularization = L12Smooth()
    loss_distill = CrossEntropyLoss()
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # Setup directory for saving equations and visualizations
    base_folder = 'ppoeql_ocatari_videos'
    os.makedirs(base_folder, exist_ok=True)
    run_folder = os.path.join(base_folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    test_folder = os.path.join(run_folder, 'test')
    record_folder = os.path.join(run_folder, 'record')
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(record_folder, exist_ok=True)
    equations_folder = "equations"
    os.makedirs(equations_folder, exist_ok=True)

    with tqdm(total=num_updates, desc="Training Progress") as pbar:
        for update in range(1, num_updates + 1): # by default ~10000 updates
            u_rate = update/num_updates
            if update%int(num_updates/100) ==0 or update==1:
                if args.ng:
                    action_func = lambda t: agent.get_action_and_value(t, threshold=args.threshold, actor="eql")[0]
                    eql_returns, eql_lengths = eval_policy(
                        envs_eval, action_func, device=device)
                    writer.add_scalar(
                        "charts/eql_returns", eql_returns, global_step)
                    writer.add_scalar(
                        "charts/eql_lengths", eql_lengths, global_step)
            if update%int(num_updates/10) ==0 or update==1:
                video_path = visual_for_ocatari_agent_videos(envs_eval, agent, device, args, test_folder, actor="neural")
                wandb.log({"test_seg": wandb.Video(video_path, fps=20, format="mp4")})

                if args.save:
                    torch.save(agent, 'models/agents/'+run_name+'.pth')

            # Annealing
            frac = 1.0 - (update - 1.0) / num_updates
            if args.anneal_lr:
                lrnow = frac * args.learning_rate
            else:
                lrnow = args.learning_rate
            if args.clip_drop:
                clip_coef_now = frac * args.clip_coef
            else:
                clip_coef_now = args.clip_coef
            if args.reg_weight_drop:
                completed_ratio = (update - 1.0) / num_updates
                reg_weight_now = args.reg_weight * completed_ratio
            else:
                reg_weight_now = args.reg_weight
            optimizer.param_groups[0]["lr"] = lrnow
            optimizer.param_groups[1]["lr"] = lrnow
            optimizer.param_groups[2]["lr"] = lrnow

            

            for step in range(0, args.num_steps):  # Individual steps (executed in parallel for 8 envs)
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    if args.pre_train and update<args.pre_train_uptates:
                        action, logprob, _, value,_,_ = agent.get_pretrained_action_and_value(next_obs)
                    else:
                        action, logprob, _, value,_,_ = agent.get_action_and_value(next_obs, threshold=args.threshold)
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob

                start_time = time.time()
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
                end_time = time.time()
                execution_time += end_time - start_time

                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                if "final_info" in info:
                    episode_return = 0.
                    episode_length = 0.
                    n_episode = 0 
                    for env_id, env_info in enumerate(info["final_info"]):
                        if not env_info is None:
                            if "episode" in env_info:
                                episode_return += env_info["episode"]["r"]
                                episode_length += env_info["episode"]["l"]
                                n_episode += 1.
                    if n_episode > 0:
                        episode_return /= n_episode
                        episode_length /= n_episode
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", episode_length, global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_actions = b_actions.long()
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            #print("Step cumulative execution time:", execution_time)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size) # 128*8=1024 by default
            clipfracs = []
            writer.add_scalar("charts/adv_mean", b_advantages.mean(), global_step)
            writer.add_scalar("charts/adv_std", b_advantages.std(), global_step)
            for epoch in range(args.update_epochs): # 4 by default
                np.random.shuffle(b_inds) # shuffling the order of obs/actions... within batch
                for start in range(0, args.batch_size, args.minibatch_size): # for minibatch in batch (4 minibatches, each one containing 32*8=256 experiences)
                    # import time
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    # minibatched evaluation of neural agent. 32 observations in a minibatch * 8 envs = 256 observations (each containing last 4 frames)
                    _, newlogprob, entropy, newvalue, newlogits, newprob = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], threshold=args.threshold)                    

                    # minibatched evaluation of eql agent
                    _, _, _, _, eq_logits, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], threshold=args.threshold, actor="eql")
                    

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > clip_coef_now).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef_now, 1 + clip_coef_now)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -clip_coef_now,
                            clip_coef_now,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    if epoch == args.update_epochs - 1:
                        if args.ng:
                            distillation_loss = loss_distill(
                                eq_logits,
                                torch.argmax(newlogits,dim=-1).long())
                            reg_loss = regularization(agent.eql_actor.get_weights_tensor())
                            writer.add_scalar("losses/reg_policy_loss", reg_loss, global_step)
                            writer.add_scalar("losses/distill_policy_loss", distillation_loss, global_step)
                        else:
                            distillation_loss = 0
                            reg_loss = 0
                    else:
                        distillation_loss = 0
                        reg_loss = 0
                        loss_cnn = 0
                    loss = pg_loss - args.ent_coef * entropy_loss\
                           + args.vf_coef * v_loss\
                           + args.distillation_loss_weight * distillation_loss\
                           + reg_weight_now * reg_loss
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    writer.add_scalar("charts/ppo_grad_norm", grad_norm, global_step)
                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break
    
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            
            writer.add_scalar("charts/update", update, global_step)
            pbar.update(1)   
    if args.save:
        torch.save(agent, 'models/agents/'+run_name+'.pth')

    # finally: collect equations and record final video of eql-agent
    video_path = visual_for_ocatari_agent_videos(envs_eval, agent, device, args, record_folder, actor="eql", n_step=2000, label="final")
    wandb.log({"final_eql": wandb.Video(video_path, fps=20, format="mp4")})

    if isinstance(agent, AgentSimplified):
        variable_names = envs.get_variable_names()
        output_names = envs.get_action_names()
        equation_list = agent.eql_actor.pretty_print(variable_names, output_names)
        save_equations(equation_list, equations_folder, run_name)

    envs.close()
    writer.close()
