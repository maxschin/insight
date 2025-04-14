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
from functools import partial
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from itertools import chain
from agents.eql.regularization import L12Smooth
from agents.agent import Agent, AgentSimplified
import matplotlib.pyplot as plt

#dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from train_cnn import CustomImageDataset, coordinate_label_to_existence_label, binary_focal_loss_with_logits
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.nn.functional as F

import copy
from visualize_utils import visual_for_ocatari_agent_videos, visual_for_agent_videos
from tqdm import tqdm
import itertools
import sympy as sy
from agents.eql.pretty_print import extract_equations
from hackatari_env import SyncVectorEnvWrapper, HackAtariWrapper
from hackatari_utils import get_reward_func_path, save_equations
from rtpt import RTPT

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=False,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Hackatari specific arguments
    parser.add_argument("-g", "--game", type=str,
                        default="PongNoFrameskip-v4", help="Game to be run")
    parser.add_argument("-m", "--modifs", nargs="+", default=[],
                        help="List of modifications to the game")
    parser.add_argument("-rf", "--reward_function", type=str,
                        default="", help="Custom reward function file name")

    # Algorithm specific arguments
    parser.add_argument("--agent_type", type=str, default="Agent", help="class name of the agent to be used")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
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
    parser.add_argument("--reg_weight", type=float, default=1e-3,
        help="regulization for interpertable")
    parser.add_argument("--use_nn", type=lambda x: bool(strtobool(x)), default=True,
        help="use nn for critic")
    parser.add_argument("--cnn_out_dim", type=int, default=128,
        help="cnn_out_dim")
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
    parser.add_argument("--load_cnn", type=lambda x: bool(strtobool(x)), default=True,
        help="load_cnn")
    parser.add_argument("--cover_cnn", type=lambda x: bool(strtobool(x)), default=False,
        help="load_cnn and cover loaded neural agent")
    parser.add_argument("--ng", type=lambda x: bool(strtobool(x)), default=True,
        help="neural guided or not")
    parser.add_argument("--fix_cnn", type=lambda x: bool(strtobool(x)), default=False,
        help="fix_cnn")
    parser.add_argument("--visual", type=lambda x: bool(strtobool(x)), default=False,
        help="visualize or not")
    parser.add_argument("--save", type=lambda x: bool(strtobool(x)), default=True,
        help="save")
    parser.add_argument("--cnn_lr_drop", type=int, default=1,
        help="cnn_lr")
    parser.add_argument("--sam_track_data", type=lambda x: bool(strtobool(x)), default=True,
        help="use dataset generated by sam_track to train the agent")
    parser.add_argument("--mass_centri_cnn", type=bool, default=False,
        help="use mass_centri_cnn or not")
    parser.add_argument("--n_objects", type=int, default=256,
        help="n_objects")
    parser.add_argument("--resolution", type=int, default=84,
        help="resolution")
    parser.add_argument("--single_frame", type=bool, default=False,
        help="single frame or not")
    parser.add_argument("--cors", type=lambda x: bool(strtobool(x)), default=True,
        help="use cors")
    parser.add_argument("--bbox", type=lambda x: bool(strtobool(x)), default=False,
        help="use bbox")
    parser.add_argument("--rgb", type=lambda x: bool(strtobool(x)), default=False,
        help="use rgb")
    parser.add_argument("--obj_vec_length", type=int, default=2,
        help="obj vector length")
    parser.add_argument("--pre_train", type=lambda x: bool(strtobool(x)), default=False,
        help="pretrain agent or not")
    parser.add_argument("--pre_train_uptates", type=int, default=500,
        help="number of pre-train update")
    parser.add_argument("--gray", type=lambda x: bool(strtobool(x)), default=True,
        help="use gray or not")
    parser.add_argument("--clip_drop", type=lambda x: bool(strtobool(x)), default=False,
        help="drop clip-coef or not")
    parser.add_argument("--pnn_guide", type=lambda x: bool(strtobool(x)), default=False,
        help="use pure nn guide or not")
    parser.add_argument("--cnn_loss_weight", type=float, default=2.)
    parser.add_argument("--coordinate_loss", type=str, default="l1")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cnn_weight_decay", type=float, default=1e-4)
    parser.add_argument("--distillation_loss_weight", type=float, default=1)
    parser.add_argument("--reg_weight_drop", type=lambda x: bool(strtobool(x)), default=True,
        help="drop reg weight or not")   

    # eql equation-specific arguments
    parser.add_argument("--equation_accuracy", type=float, default=0.01, help="The decimal point accuracy the coefficients of the eql equations should be rounded to before printing")
    parser.add_argument("--equation_threshold", type=float, default=0.05, help="Coeffecients below threshold will be filtered from eql equations before printing")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.obj_vec_length = args.cors*2+args.bbox*4+args.rgb*3
    args.cnn_out_dim = args.n_objects*args.obj_vec_length*4
    # fmt: on
    return args
    


def make_env(game, seed, args, rewardfunc_path, modifs=[]):
    def thunk():
        env = HackAtariWrapper(
            game,
            rewardfunc_path=rewardfunc_path,
            obs_mode="ori",
            frameskip=1,
            modifs=modifs
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (args.resolution, args.resolution))
        if args.gray:
            env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def eval_policy(envs, action_func, device="cuda", n_episode=10):
    obs, _ = envs.reset()
    episode_returns = []
    episode_lengths = []
    
    while len(episode_returns) < n_episode:
        # Convert observations using from_numpy for efficiency.
        obs_tensor = torch.from_numpy(obs).to(device)
        with torch.no_grad():
            action = action_func(obs_tensor)
        
        # Gymnasium's step returns: (obs, reward, terminated, truncated, infos)
        obs, rewards, terminated, truncated, infos = envs.step(action.cpu().numpy())
        
        # In your case, infos is a dict with arrays. We use the '_episode' key as a mask.
        if isinstance(infos, dict) and "_episode" in infos:
            # _episode is an array of booleans for each environment.
            done_mask = infos["_episode"]
            # Get indices of environments where the episode has finished.
            done_indices = np.nonzero(done_mask)[0]
            if done_indices.size > 0:
                # Use vectorized extraction from the episode info arrays.
                ep_returns = infos["episode"]["r"][done_indices]
                ep_lengths = infos["episode"]["l"][done_indices]
                # Extend our lists with the new finished episodes.
                episode_returns.extend(ep_returns.tolist())
                episode_lengths.extend(ep_lengths.tolist())
    
    # Compute averages for the first n_episode episodes.
    return np.mean(episode_returns[:n_episode]), np.mean(episode_lengths[:n_episode])



if __name__ == "__main__":
    # parse args
    args = parse_args()
    # run name for mostly anything related to storage
    if args.run_name == None:
        run_name = f"{args.game}" + f"_{args.agent_type}" 
        if args.reward_function:
            run_name += f"_{args.reward_function}"
        if args.modifs:
            run_name += f"_{''.join(args.modifs)}"
    else:
        run_name = args.run_name
    print("RUN NAME: " + run_name)

    # check if running inside container
    containerized = os.environ.get("container") == "podman"
    if containerized:
        print("Running inside container")
    else:
        args.total_timesteps = args.batch_size
        print("Running locally")

    ##sam_track_data:
    print("Loading CNN test data...")
    import warnings
    warnings.warn("Location is hardcoded, needs to be updated for other games than Pong!")

    asset_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "cleanrl/sam_track/assets/Pong_input")
    images_dir = os.path.join(asset_dir, 'Pong_input_masks_train')
    labels = os.path.join(images_dir, 'labels_ocatari.json')
    images_dir_test = os.path.join(asset_dir, "Pong_input_masks_test")
    labels_test = os.path.join(images_dir_test, 'labels_ocatari.json')

    train_dataset = CustomImageDataset(images_dir,labels,args,train_flag=True)
    test_dataset = CustomImageDataset(images_dir_test,labels_test,args,train_flag=True)
    train_loader = DataLoader(train_dataset, batch_size=args.minibatch_size,num_workers=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.minibatch_size,num_workers=2)
    train_data = itertools.cycle(train_loader)
    test_data = iter(test_loader)
    print("Finished loading CNN test data...")


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # maximize CPU utilization
    n_cores = len(os.sched_getaffinity(0))
    print(f"Running on: {n_cores} cores")
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(n_cores)

    # set and check device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device.type}")


    # env setup
    rewardfunc_path = get_reward_func_path(args.game, args.reward_function) if args.reward_function else None
    envs = SyncVectorEnvWrapper(
        [make_env(args.game, args.seed + i, args, rewardfunc_path) for i in range(args.num_envs)])
    envs_eval = SyncVectorEnvWrapper(
        [make_env(args.game, args.seed + i, args, rewardfunc_path) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    n_funcs=args.n_funcs
    n_actions = envs.single_action_space.n
    if args.pre_nn_agent:
        if args.pnn_guide:
            print('pnn_guide')
            agent_nn = torch.load('models/agents/'+f"{args.env_id}"+f'_{args.obj_vec_length}'+"_NN"+"_gray"*args.gray+f"_objs{args.n_objects}"+f"_seed{args.seed}"+'.pth').to(device)
        else:
            agent_nn = torch.load('models/agents/'+f"{args.env_id}"+f'_{args.obj_vec_length}'+f"_gray{args.gray}"+f"_t{args.total_timesteps}"+f"_objs{args.n_objects}"+f"_seed{args.seed}"+'.pth').to(device)
            print('nn_guide')
        agent = Agent(args,nnagent=agent_nn, n_actions=n_actions, n_funcs=n_funcs, device=device).to(device)
        for param in agent_nn.parameters():
            param.requires_grad = False
        if args.cover_cnn:
            agent.network = torch.load('models/'+f'{args.env_id}'+f'{args.resolution}'+f'{args.obj_vec_length}'+f"_gray{args.gray}"+f"_objs{args.n_objects}"+f"_seed{args.seed}"+'_od.pkl')
    else:
        agent = Agent(args, n_funcs=n_funcs, n_actions=n_actions, device=device).to(device)

    #fix hypara
    if args.fix_cnn:
            for param in agent.network.parameters():
                param.requires_grad = False
    if args.fix_cri:
            for param in agent.critic.parameters():
                param.requires_grad = False

    optimizer = optim.Adam(
        [{'params':agent.neural_actor.parameters(),'lr':args.learning_rate },
         {'params':agent.eql_actor.parameters(),'lr':args.learning_rate},
         {'params':agent.critic.parameters(),'lr':args.learning_rate},
         {'params':agent.network.parameters(),'lr':args.learning_rate/args.cnn_lr_drop,'weight_decay': args.cnn_weight_decay}], eps=1e-5)

    if args.coordinate_loss == "l2":
        coordinate_loss_fn = torch.nn.functional.mse_loss
    elif args.coordinate_loss == "l1":
        coordinate_loss_fn = torch.nn.functional.l1_loss
    else:
        raise NotImplementedError

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
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # Setup directories for logging, videos, equations
    base_folder = 'ppoeql_ocatari_videos'
    os.makedirs(base_folder, exist_ok=True)
    run_folder = os.path.join(base_folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    test_folder = os.path.join(run_folder, 'test')
    record_folder = os.path.join(run_folder, 'record')
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(record_folder, exist_ok=True)
    # equations
    equations_folder = "equations"
    os.makedirs(equations_folder, exist_ok=True)
    # model checkpoints
    os.makedirs("models/agents", exist_ok=True)
    ckpt_dir = os.path.abspath("models/agents")
    # tensorboard logs
    tensorboard_log_dir = f"runs/{run_name}"
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    # set up tensor board logging
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    with torch.no_grad():
        agent.network.train()

    # set up RTPT tracking
    rtpt = RTPT(name_initials="MS", experiment_name="INSIGHT", max_iterations=max(num_updates,1))
    rtpt.start()

    with tqdm(total=num_updates, desc="Training Progress") as pbar:
        for update in range(1, num_updates + 1):
            u_rate = update/num_updates
            if update % max(int(num_updates/100), 1) ==0 or update==1:
                acc = 0
                agent.network.eval()
                with torch.no_grad():
                    for idx, (test_x, test_label, _, _) in enumerate(test_loader):
                        test_x = test_x.to(device)
                        test_label = test_label.to(device)
                        existence_label, existence_mask = coordinate_label_to_existence_label(test_label)
                        predict_y = agent.network(test_x.float(), threshold=0.5).detach()
                        acc = acc + (coordinate_loss_fn(predict_y, test_label, reduction='none') * existence_mask).sum(1).mean(0)
                if args.ng:
                    action_func = lambda t: agent.get_action_and_value(t, threshold=args.threshold, actor="eql")[0]
                    eql_returns, eql_lengths = eval_policy(
                        envs_eval, action_func, device=device)
                    writer.add_scalar(
                        "charts/eql_returns", eql_returns, global_step)
                    writer.add_scalar(
                        "charts/eql_lengths", eql_lengths, global_step)
                agent.network.train()
                writer.add_scalar("losses/test_cnn_dataset_loss", acc/len(test_loader), global_step)
            if update%max(int(num_updates/10), 1) ==0 or update==1:
                video_path = visual_for_agent_videos(envs_eval, agent, next_obs, device, args,run_name, test_folder, threshold=args.threshold)
                if args.save:
                    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_train.pth")
                    torch.save(agent, ckpt_path)

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
            optimizer.param_groups[3]["lr"] = lrnow/args.cnn_lr_drop

            for step in range(0, args.num_steps):
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


                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
                
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                if "episode" in info and "_episode" in info:
                    # Use the '_episode' boolean array as a mask to select finished episodes.
                    done_mask = info["_episode"]
                    done_indices = np.nonzero(done_mask)[0]
                    if done_indices.size > 0:
                        ep_returns = info["episode"]["r"][done_indices]
                        ep_lengths = info["episode"]["l"][done_indices]
                        writer.add_scalar("charts/episodic_return", np.mean(ep_returns), global_step)
                        writer.add_scalar("charts/episodic_length", np.mean(ep_lengths), global_step)

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

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            writer.add_scalar("charts/adv_mean", b_advantages.mean(), global_step)
            writer.add_scalar("charts/adv_std", b_advantages.std(), global_step)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    # import time
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue,newlogits,newprob = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], threshold=args.threshold)
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
                    
                        train_x, train_label, train_label_weight, train_shape = next(train_data)
                        train_x = train_x.to(device)
                        train_label = train_label.to(device)
                        train_label_weight = train_label_weight.to(device)
                        train_shape = train_shape.to(device)
                        existence_label, existence_mask = coordinate_label_to_existence_label(train_label)
                        train_label_weight_mask = train_label_weight.unsqueeze(-1).repeat(1, 1, args.obj_vec_length).flatten(start_dim=1)
                        predict_y, existence_logits, predict_shape = agent.network(
                            train_x.float(), return_existence_logits=True, clip_coordinates=False, return_shape=True)
                        coordinate_loss = (coordinate_loss_fn(predict_y, train_label, reduction='none') * existence_mask * train_label_weight_mask).sum(1).mean(0)
                        shape_loss = (coordinate_loss_fn(predict_shape, train_shape, reduction='none') * existence_mask * train_label_weight_mask).sum(1).mean(0)
                        existence_loss = binary_focal_loss_with_logits(existence_logits, existence_label, reduction='none')
                        existence_loss = (existence_loss * train_label_weight).sum(1).mean(0)
                        loss_cnn = coordinate_loss + existence_loss + shape_loss
                        writer.add_scalar("losses/cnn_dataset_loss", loss_cnn, global_step)
                    else:
                        distillation_loss = 0
                        reg_loss = 0
                        loss_cnn = 0
                    loss = pg_loss - args.ent_coef * entropy_loss\
                           + args.vf_coef * v_loss\
                           + args.cnn_loss_weight * loss_cnn\
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
            rtpt.step()

    if args.save:
        print("Saving final checkpoint...")
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_final.pth")
        torch.save(agent, ckpt_path)

    # record eql and neural agent
    print("Recording agents...")
    visual_for_ocatari_agent_videos(envs_eval, agent, device, args, record_folder, actor="eql", n_step=1000, label="final")
    visual_for_ocatari_agent_videos(envs_eval, agent, device, args, record_folder, actor="neural", n_step=1000, label="final")

    # save equations
    print("Saving equations...")
    print("ATTENTION: HARDCODED STUFF!!!")
    var_names = envs.get_variable_names_hardcoded_pong()
    output_names = ["NOOP_1", "NOOP_2", "UP_1", "DOWN_1", "UP_2", "DOWN_2"]
    equations, _ = extract_equations(
        agent,
        var_names,
        output_names,
        accuracy=args.equation_accuracy,
        threshold=args.equation_threshold,
        use_multiprocessing=True
    )
    save_equations(equations, equations_folder, run_name)

    envs.close()
    writer.close()
