import os
import sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(SRC)

import argparse
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, ProgressBarCallback, EveryNTimesteps
from stable_baselines3 import PPO
from typing import Callable
import torch
from torch.nn.modules import container
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from rtpt import RTPT

from agents.agent import load_agent
from imitation_learning.utils import fill_replay_buffer, DeterministicReplayBuffer
from utils.hackatari_env import HackAtariWrapper, SyncVectorEnvWrapper
from utils.utils import save_equations, get_reward_func_path, make_env
from agents.eql.regularization import L12Smooth
from utils.visualize_utils import visual_for_ocatari_agent_videos
from utils.callbacks import RtptCallback

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument("--run-name", type=str, default=None,
        help="the defined run_name")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")

    # Hackatari specific arguments
    parser.add_argument("-g", "--game", type=str,
                        default="Pong", help="Game to be run")
    parser.add_argument("-m", "--modifs", nargs="+", default=[],
                        help="List of modifications to the game")
    parser.add_argument("-rf", "--reward_function", type=str,
                        default="", help="Custom reward function file name")

    # agent specific args
    parser.add_argument("--agent_type", type=str, default="Agent", help="class name of the agent to be used")
    return parser.parse_args()


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

if __name__ == "__main__":
    # handle args
    args = parse_args()
    # run name for mostly anything related to storage
    if args.run_name == None:
        run_name = f"{args.game}" + f"_{args.agent_type}" 
        if args.reward_function:
            run_name += f"_{args.reward_function}"
        if args.modifs:
            run_name += f"_{''.join(args.modifs)}"
        run_name += "_oc" # denotes object-centric agent
    else:
        run_name = args.run_name
    args.n_funcs, args.n_layers, args.deter_action, args.threshold = 0,0, False, 0 # meaningless

    ########### DEFINE HYPERPARAMS ###########################
    # check if running inside container
    containerized = os.environ.get("container") == "podman"
    if containerized:
        print("Running inside container")
    else:
        print("Running locally")

    # parameters for training
    adam_step_size = 0.00025
    clipping_eps = 0.1
    training_timesteps = 25_000_000 if containerized else 100
    verbose=0

    # for evaluation
    n_cores = len(os.sched_getaffinity(0))
    args.num_envs = n_cores
    print(f"Running on: {n_cores} cores")
    n_eval_episodes = args.num_envs // 2 
    eval_frequency = 500_000 if containerized else 50

    # params for distillation
    replay_capacity = 1_000_000 if containerized else 50
    replay_priority_weight = 0.6
    reg_weight = 1e-3 / 2
    num_distillation_epochs = 150 if containerized else 3
    distillation_eval_freq = 20
    batch_size = 1_024
    n_eval_episodes_eql_distillation = 100

    # for recording videos
    recording_timesteps = 1_000 if containerized else 10

    # rtpt
    rtpt_frequency = 10_000 if containerized else 10

    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(n_cores)

    ##########################################################

    # create rtpt for process tracking on remote server
    max_iterations = (training_timesteps // rtpt_frequency) + num_distillation_epochs
    rtpt = RTPT(name_initials="MS", experiment_name="INSIGHT", max_iterations=max_iterations)
    rtpt.start()

    # Setup directories for logging, videos, equations
    base_folder = os.path.join(SRC, 'ppoeql_ocatari_videos')
    os.makedirs(base_folder, exist_ok=True)
    run_folder = os.path.join(base_folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    test_folder = os.path.join(run_folder, 'test')
    record_folder = os.path.join(run_folder, 'record')
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(record_folder, exist_ok=True)
    # equations
    equations_folder = os.path.join(SRC, "equations")
    os.makedirs(equations_folder, exist_ok=True)
    # model checkpoints
    os.makedirs(os.path.join(SRC, "models/agents"), exist_ok=True)
    ckpt_dir = os.path.abspath(os.path.join(SRC, "models/agents"))
    # tensorboard logs
    tensorboard_log_dir = os.path.join(SRC, f"runs/{run_name}")
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    ############### INITIAL TRAINING OF TEACHER ##############

    # Setup multiple HackAtari environments
    rewardfunc_path = get_reward_func_path(args.game, args.reward_function) if args.reward_function else None
    env = SubprocVecEnv([make_env(args.game, args.seed + i, modifs=args.modifs, rewardfunc_path=rewardfunc_path, sb3=True) for i in range(args.num_envs)], start_method="fork")
    env_eval = SubprocVecEnv([make_env(args.game, args.seed + i, modifs=args.modifs, rewardfunc_path=rewardfunc_path, sb3=True) for i in range(args.num_envs // 2)], start_method="fork")

    # load agent wrapped correctly for SB3 PPO training
    CustomAgent = load_agent(args.agent_type, for_sb3=True)

    # set up callbacks for training
    # evaluations
    eval_cb = EvalCallback(
        env_eval,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=None,
        log_path=None,
        eval_freq=max(eval_frequency // args.num_envs , 1),
        deterministic=True,
        render=False
    )
    # for progress bar
    tqdm_cb = ProgressBarCallback()

    # for RTPT updating
    rtpt_callback = RtptCallback(rtpt=rtpt)
    n_callback = EveryNTimesteps(
            n_steps=rtpt_frequency,
            callback=rtpt_callback
    )
    cb_list = CallbackList([tqdm_cb, eval_cb])


    # Set up PPO
    model = PPO(
        policy=lambda obs_space, act_space, lr, use_sde: CustomAgent(obs_space, act_space, lr, args, env),
        n_steps=128,
        learning_rate=linear_schedule(adam_step_size),
        n_epochs=3,
        batch_size=32*8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=linear_schedule(clipping_eps),
        vf_coef=1,
        ent_coef=0.01,
        env=env,
        verbose=0,
        tensorboard_log=tensorboard_log_dir,
        device="cpu"
    )

    # train agent
    model.learn(
        total_timesteps=training_timesteps,
        callback=cb_list
    )

    # save agent
    print("Saving checkpoint...")
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_neural.pth")
    torch.save(model.policy.agent, ckpt_path)

    ####### DISTILLATION PHASE ####################
    # distill agent by imitation learning
    # Here we perform imitation/distillation learning: we train eql_actor (student)
    # to mimic the neural_actor (teacher) using a cross-entropy divegence loss.

    # instantiate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # collect samples for replay buffer
    print("Collecting samples for imitation learning...")
    buffer = DeterministicReplayBuffer(capacity=replay_capacity)
    device="cpu"
    agent_path = os.path.join(ckpt_dir, f"{run_name}_neural.pth")
    agent = model.policy.agent
    agent.to(device)
    fill_replay_buffer(env, agent, buffer, device, target_size=replay_capacity)


    # Set teacher (neural_actor) to evaluation mode and student (eql_actor) to train mode.
    agent.neural_actor.eval()
    agent.eql_actor.train()

    # Create an optimizer for the student network only.
    student_optimizer = optim.Adam(agent.eql_actor.parameters(), lr=1e-3)
    regularization = L12Smooth()
    cross_ent_loss = CrossEntropyLoss()
    reg_weight_now = reg_weight

    # create a summary writer to track loss
    writer = SummaryWriter(log_dir=model.logger.dir)

    # the data loader for loading from replay buffer
    dataloader = DataLoader(buffer, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # distill
    print("Starting distillation phase...")
    progress_bar = tqdm(range(num_distillation_epochs), desc="Distillation phase", unit="epoch")

    batch_num = 0
    for epoch in progress_bar:
        agent.eql_actor.train()
        for batch_obs, batch_teacher_action in dataloader:
            # Move batch data to device
            batch_obs = batch_obs.to(device)
            batch_teacher_action = batch_teacher_action.to(device)

            # Get student probabilities from the 'eql' actor network.
            # Assuming agent.get_action_and_value returns a tuple with student_probs as the 6th element.
            _, _, _, _, _, student_logits = agent.get_action_and_value(batch_obs, actor="eql")

            # use cross entropy loss
            imitation_loss = cross_ent_loss(
                    student_logits,
                    batch_teacher_action
            )
            
            # Compute regularization loss.
            reg_loss = regularization(agent.eql_actor.get_weights_tensor())
            loss = imitation_loss + reg_weight_now * reg_loss

            # Update student network parameters.
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()

            # Optionally, you can log batch-level metrics here:
            writer.add_scalar("Distillation/Loss", loss.item(), batch_num)
            writer.add_scalar("Distillation/ImitationLoss", imitation_loss.item(), batch_num)
            writer.add_scalar("Distillation/RegLoss", reg_loss.item(), batch_num)
            batch_num += 1

        # Update progress tracking
        if epoch % rtpt_frequency == 0:
            rtpt.step()

        # eval eql agent on environments
        if epoch % distillation_eval_freq == 0 or epoch == 0:
            agent.eql_actor.eval()
            action_func = lambda t: agent.get_action_and_value(t, threshold=args.threshold, actor="eql")[0]
            eql_returns, eql_lengths = eval_policy(
                        env_eval, action_func, device=device)
            writer.add_scalar("Distillation/EQL_returns", eql_returns, epoch)
            writer.add_scalar("Distillation/EQL_lengths", eql_lengths, epoch)

        # Update regularization weight to increase linearly from zero to full weight over epochs.
        reg_weight_now = ((epoch / num_distillation_epochs) ** 2) * reg_weight

    # save equations
    print("Saving equations...")
    # TODO

    # save agent
    print("Saving checkpoint...")
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_final.pth")
    torch.save(agent, ckpt_path)
    
    # record eql and neural agent
    print("Recording agents...")
    device = "cpu"
    visual_for_ocatari_agent_videos(env_eval, agent, device, args, record_folder, actor="eql", n_step=recording_timesteps, label="final")
    visual_for_ocatari_agent_videos(env_eval, agent, device, args, record_folder, actor="neural", n_step=recording_timesteps, label="final")

    writer.close()
    print(f"Finished run: {run_name}")


