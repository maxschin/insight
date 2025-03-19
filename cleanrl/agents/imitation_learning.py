import random
import torch
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # Stores priorities
        self.alpha = alpha  # Controls priority strength

    def push(self, observation, teacher_probs, priority=1.0):
        """Stores an (observation, teacher_probs) tuple with an initial priority."""
        self.buffer.append((observation, teacher_probs))
        priority = max(priority, 1e-5)
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        """Samples a batch using priority-based probabilities."""
        if len(self.buffer) == 0:
            raise ValueError("Replay buffer is empty!")

        # Compute sampling probabilities
        priorities_np = np.array(self.priorities, dtype=np.float32)
        probs = priorities_np ** self.alpha  # Priorities raised to alpha
        probs /= probs.sum()  # Normalize to get valid probabilities

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling weights
        weights = (1.0 / (len(self.buffer) * probs[indices])) ** beta
        weights /= weights.max()  # Normalize for stability

        obs, targets = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            indices,
        )

    def update_priorities(self, indices, losses):
        """Updates priorities based on the observed training losses."""
        for idx, loss in zip(indices, losses):
            self.priorities[idx] = max(loss.item() + 1e-5, 1e-5)  # Avoid zero priority

    def __len__(self):
        return len(self.buffer)

def fill_replay_buffer(env, agent, buffer, device, target_size=10000):
    """
    Efficiently fills the replay buffer using a vectorized environment (SubprocVecEnv).
    Uses the trained agent's teacher network to store (obs, teacher_probs) pairs.
    
    :param env: Vectorized Gym environment (SubprocVecEnv)
    :param agent: Trained agent (with neural_actor as teacher)
    :param buffer: PrioritizedReplayBuffer instance
    :param target_size: Maximum buffer size to fill
    """
    print(f"Filling replay buffer until it reaches {target_size} samples...")
    
    obs = env.reset()
    if isinstance(obs, tuple):  # Handle vectorized envs that return (obs, info)
        obs = obs[0]

    with torch.no_grad():
        while len(buffer) < target_size:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            # Get teacher's action probabilities for current observations
            action ,_,_,_,_, teacher_probs = agent.get_action_and_value(obs_tensor, actor="neural")

            teacher_probs = teacher_probs.cpu().numpy()

            # Store all (obs, teacher_probs) pairs in the buffer
            for o, tp in zip(obs, teacher_probs):
                buffer.push(o, tp)  # Default priority = 1.0 initially

            # Use action from policy
            action = action.cpu().numpy()
            obs, _, _, _ = env.step(action)

            if isinstance(obs, tuple):
                obs = obs[0]  # Extract actual observation data from tuple

    print(f"Replay buffer filled with {len(buffer)} samples.")
