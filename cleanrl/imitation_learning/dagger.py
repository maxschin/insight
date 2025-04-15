import os
import sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(SRC)

import torch
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter
from imitation_learning.utils import collect_rollouts_eql, collect_training_targets_neural, DeterministicReplayBuffer
from utils.utils import eval_policy, make_env
from typing import Callable
from agents.eql.regularization import L12Smooth
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv

regularization = L12Smooth()
cross_ent_loss = CrossEntropyLoss()

class Dagger:
    def __init__(
            self, 
            env,
            agent,
            optimizer,
            dagger_iter = 1_000,
            dagger_iter_size = 102_400,
            epochs_per_iter = 100,
            mini_batch_size = 1024,
            learning_rate = 1e-3,
            regularization_rate = 1e-3,
            distillation_loss = CrossEntropyLoss(),
            regularization_loss = L12Smooth(),
            max_agg_data = 500_000,
            device = "cpu",
            n_eval_episodes = 10
    ):
        self.env = env
        self.agent = agent
        self.optim = optimizer
        self._learning_rate = learning_rate
        self._reg_rate = regularization_rate
        self.distillation_loss = distillation_loss
        self.regularization_loss = regularization_loss
        self.dagger_iter = dagger_iter
        self.dagger_iter_size = dagger_iter_size
        self.epochs_per_iter = epochs_per_iter
        self.mini_batch_size = mini_batch_size
        self.n_eval_episodes = n_eval_episodes

        self.cur_dagger_iter = 0
        self.progress_remaining = self.cur_dagger_iter / self.dagger_iter
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.replay_buffer = DeterministicReplayBuffer(capacity=max_agg_data)

    def imitation_learn(self, summary_writer, rtpt=None):
        # set in train/eval
        self.agent.eql_actor.train()
        self.agent.network.eval()
        self.agent.critic.eval()
        self.agent.neural_actor.eval()

        # global iterations
        update_step = 0
        progress_bar = tqdm(range(self.dagger_iter), desc="Dagger distill...", unit="iter")
        for dagger_iter in progress_bar:
            # collect new student observations, predictions
            with torch.no_grad():
                new_obs,_ = collect_rollouts_eql(
                        env=self.env,
                        agent=self.agent,
                        device=self.device,
                        n=self.dagger_iter_size
                )
                # collect teacher actions to go with it
                neural_actions,_ = collect_training_targets_neural(
                        observations=new_obs,
                        agent=self.agent,
                        device=self.device
                )
             # add to replay buffer
            new_obs_np = new_obs.cpu().numpy()
            neural_actions_np = neural_actions.cpu().numpy()
            for obs_sample, action_sample in zip(new_obs_np, neural_actions_np):
                self.replay_buffer.push(obs_sample, action_sample)

            # update rates
            self.progress_remaining = self.cur_dagger_iter / self.dagger_iter
            learning_rate = self._get_learning_rate()
            reg_rate = self._get_reg_rate()
            for param_group in self.optim.param_groups:
                param_group["lr"] = learning_rate

            train_loader = DataLoader(
                dataset=self.replay_buffer,
                batch_size=self.mini_batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=min(4, len(os.sched_getaffinity(0)) // 2),
                pin_memory=(self.device.type == 'cuda')
            )

            for epoch in range(self.epochs_per_iter):
                for obs_batch, target_batch in train_loader:
                    agent.eql_actor.train()
                    obs_batch = obs_batch.to(self.device, non_blocking=True)
                    target_batch = target_batch.to(self.device, non_blocking=True)

                    self.optim.zero_grad()
                    _, _, _, _, eql_logits, _ = self.agent.get_action_and_value(obs_batch, actor="eql")

                    distill_loss = self.distillation_loss(eql_logits, target_batch)
                    reg_loss = self.regularization_loss(self.agent.eql_actor.get_weights_tensor())
                    loss = distill_loss + reg_rate * reg_loss

                    loss.backward()
                    self.optim.step()

                    # Log losses to TensorBoard
                    summary_writer.add_scalar("Rate/LearningRate", learning_rate, update_step)
                    summary_writer.add_scalar("Rate/RegularizationRate", reg_rate, update_step)
                    summary_writer.add_scalar("Loss/Distillation", distill_loss.item(), update_step)
                    summary_writer.add_scalar("Loss/Regularization", reg_loss.item(), update_step)
                    summary_writer.add_scalar("Loss/Total", loss.item(), update_step)
                    update_step += 1

            # test agents
            mean_episode_reward = evaluate_agent(self.agent, self.env, episodes=self.n_eval_episodes, actor="eql", device=self.device)
            writer.add_scalar("Distillation/EQL_returns", mean_episode_reward, dagger_iter)

            # increment counters
            self.cur_dagger_iter += 1
            if rtpt is not None:
                rtpt.step()
        pass

    def _get_learning_rate(self):
        if callable(self._learning_rate):
            return self._learning_rate(self.progress_remaining)
        return self._learning_rate

    def _get_reg_rate(self):
        if callable(self._reg_rate):
            return self._reg_rate(self.progress_remaining)
        return self._reg_rate


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def quadratic_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return (progress_remaining ** 2) * initial_value
    return func

if __name__ == "__main__":
    print("Running DAGGER distillation")
    game = "PongNoFrameskip-v4"

    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load agent
    os.makedirs(os.path.join(SRC, "models/agents"), exist_ok=True)
    ckpt_dir = os.path.abspath(os.path.join(SRC, "models/agents"))
    run_name = "PongNoFrameskip-v4_Agent"
    agent_path = os.path.join(ckpt_dir, f"{run_name}_oc_final.pth")
    agent = torch.load(agent_path, weights_only=False)
    agent.to(device)

    # set up envs
    n_cores = len(os.sched_getaffinity(0))
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(n_cores)
    env = SubprocVecEnv(
        [make_env(game, 42 + i, modifs=[], rewardfunc_path=None, sb3=False) for i in range(n_cores)], start_method="fork")

    # prepare logging
    run_name = run_name + "_Dagger"
    tensorboard_log_dir = os.path.join(SRC, f"runs/{run_name}")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # check if running inside container
    containerized = os.environ.get("container") == "podman"
    if containerized:
        print("Running inside container")
    else:
        print("Running locally")

    # classic hyperparams
    learning_rate = 1e-3
    regularization_rate = 0
    dagger_iter = 200 if containerized else 3
    dagger_iter_size = 50_000 if containerized else 1_000
    epochs_per_iter = 20 if containerized else 3
    mini_batch_size = 1024
    regularization_rate = linear_schedule(1e-5)
    max_agg_data = 500_000 if containerized else 2_000
    n_eval_episodes = 10

    # define the optim
    optimizer = optim.Adam(agent.eql_actor.parameters(), learning_rate)

    # just to be sure eval neural agent as baseline
    action_func = lambda t: agent.get_action_and_value(t, threshold=, actor="neural")[0]
    neural_returns, _ = eval_policy(
        env, action_func, device=device)
    print(f"NEURAL RETURNS: {neural_returns}")

    # the learning
    dagger = Dagger(
        env,
        agent,
        optimizer,
        learning_rate=learning_rate,
        regularization_rate=regularization_rate,
        dagger_iter=dagger_iter,
        dagger_iter_size=dagger_iter_size,
        epochs_per_iter=epochs_per_iter,
        mini_batch_size=mini_batch_size,
        n_eval_episodes=n_eval_episodes,
        max_agg_data=max_agg_data,
        device=device
    )
    print("Starting imitation learning...")
    dagger.imitation_learn(writer)

