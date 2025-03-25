import random
import torch
import numpy as np
from collections import deque
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


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

class PrioritizedReplayBufferDataset(Dataset):
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []       # List to store (observation, teacher_probs)
        self.priorities = []   # List to store corresponding priorities

    def push(self, observation, teacher_probs, priority=1.0):
        """Adds a new sample with its initial priority."""
        priority = max(priority, 1e-5)
        if len(self.buffer) < self.capacity:
            self.buffer.append((observation, teacher_probs))
            self.priorities.append(priority)
        else:
            # Overwrite oldest entry
            index = len(self.buffer) % self.capacity
            self.buffer[index] = (observation, teacher_probs)
            self.priorities[index] = priority

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        # Returns the observation and teacher probabilities.
        observation, teacher_probs = self.buffer[idx]
        return observation, teacher_probs

    def get_sampler(self, batch_size):
        """Creates a WeightedRandomSampler using current priorities."""
        # Convert priorities to probabilities (powered by alpha)
        priorities_np = np.array(self.priorities, dtype=np.float32)
        weights = priorities_np ** self.alpha
        # The WeightedRandomSampler samples indices with replacement according to the given weights.
        return WeightedRandomSampler(weights=weights, num_samples=batch_size, replacement=True)

    def update_priorities(self, indices, losses):
        """Updates priorities based on new loss values."""
        for idx, loss in zip(indices, losses):
            self.priorities[idx] = max(loss.item() + 1e-5, 1e-5)

class DeterministicReplayBuffer(Dataset):
    def __init__(self, capacity):
        """
        Creates a deterministic replay buffer.
        :param capacity: Maximum number of samples to store.
        """
        self.capacity = capacity
        self.buffer = []
        self.next_idx = 0  # Tracks where to write the next sample once capacity is reached.

    def push(self, observation, teacher_probs):
        """
        Adds a new sample to the buffer. If capacity is reached, overwrites the oldest sample.
        :param observation: The observation sample.
        :param teacher_probs: The teacher's output probabilities.
        """
        # If buffer is not full, append the sample.
        if len(self.buffer) < self.capacity:
            self.buffer.append((observation, teacher_probs))
        else:
            # Once full, overwrite in a circular manner.
            self.buffer[self.next_idx] = (observation, teacher_probs)
        # Update the index for circular overwrite.
        self.next_idx = (self.next_idx + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        """
        Retrieves a sample at a given index.
        Converts the observation and teacher_probs to torch tensors.
        """
        obs, teacher_probs = self.buffer[index]
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        teacher_probs_tensor = torch.tensor(teacher_probs, dtype=torch.long)
        return obs_tensor, teacher_probs_tensor

def fill_replay_buffer(env, agent, buffer, device, target_size=10000, random_action_pct=70):
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
            action ,_,_,_, teacher_logits,_ = agent.get_action_and_value(obs_tensor, actor="neural")

            teacher_logits = teacher_logits.cpu().numpy()
            teacher_action = np.argmax(teacher_logits, axis=-1)

            # Store all (obs, teacher_probs) pairs in the buffer
            for o, tp in zip(obs, teacher_action):
                buffer.push(o, tp)  # Default priority = 1.0 initially

            # Use action from policy in 1-random_action_pct of cases
            action = action.cpu().numpy()

            # Create a random mask to determine which actions should be random.
            # For each instance, with probability random_action_pct/100, the action is overridden.
            rand_mask = np.random.rand(len(action)) < (random_action_pct / 100.0)
            if np.any(rand_mask):
                # For discrete action spaces, vectorize sampling random actions.
                random_actions = np.random.randint(0, env.action_space.n, size=rand_mask.sum())
                action[rand_mask] = random_actions

            obs, _, _, _ = env.step(action)

            if isinstance(obs, tuple):
                obs = obs[0]  # Extract actual observation data from tuple

    print(f"Replay buffer filled with {len(buffer)} samples.")


def collect_rollouts_eql(env, agent, n, device, random_action_pct=0):
    """
    Collect rollouts using the EQL actor with gradient tracking and optional random action override.
    
    For n steps per environment, this function uses the EQL actor to select an action,
    collects the observations and the corresponding EQL logits. With probability given by
    random_action_pct (a percentage between 0 and 100), the action from the EQL actor is replaced
    by a random action sampled from the environment's action space.
    
    The returned tensors are flattened so that the batch dimension is n_steps * num_envs.
    
    :param env: Vectorized Gym environment.
    :param agent: Trained agent that supports get_action_and_value with actor="eql".
    :param n: Number of steps per environment.
    :param device: Torch device (e.g., 'cuda' or 'cpu').
    :param random_action_pct: Percentage (0-100) chance to override the EQL action with a random action.
    :return: A tuple (observations, eql_logits) as torch tensors:
             - observations: shape (n_steps * num_envs, *obs_shape)
             - eql_logits: shape (n_steps * num_envs, logits_dim)
    """
    # Reset the environment and extract the initial observation.
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    obs_list = []    # Each element: (num_envs, *obs_shape)
    logits_list = [] # Each element: (num_envs, logits_dim)
    
    for _ in range(n):
        # Convert the numpy observation to a torch tensor.
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        
        # Forward pass using the EQL actor (gradient tracking enabled).
        action, _, _, _, eq_logits, _ = agent.get_action_and_value(obs_tensor, actor="eql")
        
        # Save the current observation and corresponding logits.
        obs_list.append(obs_tensor)
        logits_list.append(eq_logits)
        
        # Use the action to step the environment.
        # Detach, move to CPU, and convert to numpy.
        action_np = action.detach().cpu().numpy()
        
        # Create a random mask: with probability (random_action_pct/100) override the action.
        rand_mask = np.random.rand(len(action_np)) < (random_action_pct / 100.0)
        if np.any(rand_mask):
            # For discrete action spaces, sample random actions using env.action_space.
            random_actions = np.random.randint(0, env.action_space.n, size=rand_mask.sum())
            action_np[rand_mask] = random_actions
        
        obs, _, _, _ = env.step(action_np)
        if isinstance(obs, tuple):
            obs = obs[0]
    
    # Flatten the collected observations and logits.
    # Each tensor in obs_list has shape (num_envs, *obs_shape) so concatenating along dim=0
    # yields a tensor of shape (n_steps * num_envs, *obs_shape)
    observations = torch.cat(obs_list, dim=0)
    eql_logits = torch.cat(logits_list, dim=0)
    
    return observations, eql_logits

def collect_training_targets_neural(observations, agent, device):
    """
    Given a batch of flattened observations, use the neural (teacher) actor to compute training targets.
    
    This function takes a tensor of observations with shape (batch_size, *obs_shape),
    passes it through the neural actor (under a no_grad context) to obtain the logits,
    and then computes the teacher actions by taking the argmax along the logits dimension.
    
    :param observations: torch.Tensor of shape (batch_size, *obs_shape)
    :param agent: Trained agent that supports get_action_and_value with actor="neural".
    :param device: Torch device.
    :return: 
             - teacher_actions: shape (batch_size,)
    """
    # Ensure the observations are on the correct device.
    obs_tensor = observations.to(device)
    
    with torch.no_grad():
        # Compute teacher's logits using the neural actor.
        _, _, _, _, teacher_logits, _ = agent.get_action_and_value(obs_tensor, actor="neural")
    
    # Compute teacher actions by taking the argmax along the logits dimension.
    teacher_actions = teacher_logits.argmax(dim=-1)
    
    return teacher_actions, teacher_logits
