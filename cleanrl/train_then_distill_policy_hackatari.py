import os
import argparse
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, ProgressBarCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from typing import Callable
import torch
from torch.nn.modules import container
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from rtpt import RTPT

from agents.agent import load_agent
from agents.imitation_learning import PrioritizedReplayBuffer, fill_replay_buffer
from hackatari_env import HackAtariWrapper, SyncVectorEnvWrapper
from hackatari_utils import save_equations, get_reward_func_path
from agents.eql.regularization import L12Smooth
from visualize_utils import visual_for_ocatari_agent_videos
from callbacks import RtptCallback


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
    parser.add_argument("--agent_type", type=str, default="AgentSimplified", help="class name of the agent to be used")
    return parser.parse_args()

def make_env(env_name, seed, rewardfunc_path, modifs, index, is_eval=False, video_folder=None):
    def thunk():
        env = HackAtariWrapper(env_name, modifs=modifs, rewardfunc_path=rewardfunc_path)  
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return thunk

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
    else:
        run_name = args.run_name
    args.n_funcs, args.n_layers, args.deter_action = 0,0, False # meaningless

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
    training_timesteps = 20_000_000 if containerized else 200
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
    reg_weight = 1e-3
    num_distillation_epochs = 200_000 if containerized else 3
    batch_size = 128

    # for recording videos
    recording_timesteps = 1_000 if containerized else 10

    # rtpt
    rtpt_frequency = 10_000 if containerized else 10

    ##########################################################

    # create rtpt for process tracking on remote server
    max_iterations = (training_timesteps + num_distillation_epochs) // rtpt_frequency
    rtpt = RTPT(name_initials="MS", experiment_name="INSIGHT", max_iterations=max_iterations)
    rtpt.start()

    # for storing and logging
    # videos
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

    ############### INITIAL TRAINING OF TEACHER ##############

    # Setup multiple HackAtari environments
    rewardfunc_path = get_reward_func_path(args.game, args.reward_function) if args.reward_function else None
    env = SubprocVecEnv([make_env(args.game, args.seed + i, modifs=args.modifs, rewardfunc_path=rewardfunc_path, index=i) for i in range(args.num_envs)], start_method="fork")
    env_eval = SubprocVecEnv([make_env(args.game, args.seed + i, modifs=args.modifs, rewardfunc_path=rewardfunc_path, index=i, is_eval=True, video_folder=record_folder) for i in range(args.num_envs // 2)], start_method="fork")

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

    ####### DISTILLATION PHASE ####################
    # distill agent by imitation learning
    # Here we perform imitation/distillation learning: we train eql_actor (student)
    # to mimic the neural_actor (teacher) using a KL divergence loss.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = model.policy.agent
    agent.to(device)


    # Set teacher (neural_actor) to evaluation mode and student (eql_actor) to train mode.
    agent.neural_actor.eval()
    agent.eql_actor.train()

    # Create an optimizer for the student network only.
    student_optimizer = optim.Adam(agent.eql_actor.parameters(), lr=1e-4)
    regularization = L12Smooth()
    reg_weight_now = reg_weight

    # create a summary writer to track loss
    writer = SummaryWriter(log_dir=model.logger.dir)
    global_step = training_timesteps

    # collect samples for replay buffer
    prioritized_replay_buffer = PrioritizedReplayBuffer(capacity=replay_capacity, alpha=replay_priority_weight)
    fill_replay_buffer(env, agent, prioritized_replay_buffer, device, target_size=replay_capacity)

    # distill
    print("Starting distillation phase...")
    progress_bar = tqdm(range(num_distillation_epochs), desc="Distillation phase", unit="epoch")
    for epoch in progress_bar:
        # Sample a batch of observations from the environment.
        batch_obs, batch_teacher_probs, batch_weights, batch_indices = prioritized_replay_buffer.sample(batch_size)
        batch_obs, batch_teacher_probs, batch_weights = (
                batch_obs.to(device),
                batch_teacher_probs.to(device),
                batch_weights.to(device),
        )

        # get eql probs for the batch of observations
        _,_,_,_,_, student_probs = agent.get_action_and_value(batch_obs, actor="eql")

        # compute distillation (with importance-sampling correction) and regularization loss
        per_sample_imitation_loss = F.kl_div(student_probs, batch_teacher_probs, reduction='none').sum(dim=1)
        imitation_loss = (per_sample_imitation_loss * batch_weights).mean()
        reg_loss = regularization(agent.eql_actor.get_weights_tensor())
        loss = imitation_loss + reg_weight_now * reg_loss

        # update parameters
        student_optimizer.zero_grad()
        loss.backward()
        student_optimizer.step()

        # update prioritized replay buffer importance weights
        prioritized_replay_buffer.update_priorities(batch_indices, per_sample_imitation_loss.detach().cpu())

        # log values
        writer.add_scalar("Distillation/Loss", loss.item(), global_step)
        writer.add_scalar("Distillation/ImitationLoss", imitation_loss.item(), global_step)
        writer.add_scalar("Distillation/RegLoss", loss.item(), global_step)
        global_step += 1

        # update rtpt
        if epoch % rtpt_frequency == 0:
            rtpt.step()

        # update reg weight to decay linearly to zero
        reg_weight_now = (epoch / num_distillation_epochs) * reg_weight

    # save equations
    print("Saving equations...")
    variable_names = env.env_method("get_variable_names", indices=[0])[0]
    output_names = env.env_method("get_action_names", indices=[0])[0]
    equation_list = agent.eql_actor.pretty_print(variable_names, output_names)
    save_equations(equation_list, equations_folder, run_name)

    # save agent
    print("Saving checkpoint...")
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_final.pth")
    torch.save(agent, ckpt_path)
    
    # record eql and neural agent
    print("Recording agents...")
    visual_for_ocatari_agent_videos(env_eval, agent, device, args, record_folder, actor="eql", n_step=recording_timesteps, label="final")
    visual_for_ocatari_agent_videos(env_eval, agent, device, args, record_folder, actor="neural", n_step=recording_timesteps, label="final")

    writer.close()
    print(f"Finished run: {run_name}")


