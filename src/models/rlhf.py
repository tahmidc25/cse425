"""
Task 4: RLHF - Reinforcement Learning from Human Feedback
Components:
1. Reward Model: LSTM-based scoring of music quality
2. Policy Gradient Trainer: REINFORCE algorithm to fine-tune generator
3. Heuristic Reward Function: Based on music features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict


class RewardModel(nn.Module):
    """
    LSTM-based reward model that scores music sequences.
    Input: piano roll (batch, seq_len, 128)
    Output: scalar reward in [0, 1]
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        """
        Args:
            input_dim: Piano roll pitch dimension (128)
            hidden_dim: LSTM hidden dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 2-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Scoring head
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Piano roll (batch, seq_len, 128)
        
        Returns:
            Reward scores (batch, 1) in [0, 1]
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        
        # Score
        hidden = F.relu(self.fc1(last_hidden))
        reward = torch.sigmoid(self.fc2(hidden))  # [0, 1]
        
        return reward


class HeuristicRewardFunction:
    """
    Heuristic reward function based on music features.
    Rewards:
    - Pitch variety (higher entropy)
    - Rhythm density (reasonable number of notes per bar)
    - Scale conformity (notes in C major or A minor pentatonic)
    - Penalties for excessive repetition
    """
    
    def __init__(self):
        """Initialize heuristic reward function."""
        # C major scale: C, D, E, F, G, A, B (C=0)
        self.c_major = {0, 2, 4, 5, 7, 9, 11}
        # A minor pentatonic: A, C, D, E, G (A=9)
        self.a_minor_pent = {9, 0, 2, 4, 7}
    
    def compute_reward(self, piano_roll: np.ndarray) -> float:
        """
        Compute reward from piano roll.
        
        Args:
            piano_roll: (seq_len, 128) array
        
        Returns:
            Reward score in [0, 1]
        """
        reward = 0.0
        
        # 1. Pitch variety (entropy)
        pitch_histogram = np.sum(piano_roll, axis=0)
        pitch_histogram = pitch_histogram / (np.sum(pitch_histogram) + 1e-6)
        pitch_entropy = -np.sum(pitch_histogram * np.log(pitch_histogram + 1e-6))
        pitch_entropy_norm = min(pitch_entropy / np.log(128), 1.0)
        reward += 0.3 * pitch_entropy_norm
        
        # 2. Rhythm density (reasonable density)
        note_density = np.sum(piano_roll > 0) / (piano_roll.shape[0] * 128)
        density_reward = 1.0 - np.abs(note_density - 0.15)  # Ideal ~15%
        reward += 0.2 * density_reward
        
        # 3. Scale conformity
        active_pitches = np.where(np.sum(piano_roll, axis=0) > 0)[0]
        if len(active_pitches) > 0:
            pitch_classes = active_pitches % 12
            scale_union = self.c_major | self.a_minor_pent
            conformity = np.sum([p in scale_union for p in pitch_classes]) / len(pitch_classes)
            reward += 0.3 * conformity
        
        # 4. Repetition penalty (penalize exact repetition)
        # Check for repeated patterns of length 4
        window_size = 4
        if piano_roll.shape[0] >= 2 * window_size:
            repetitions = 0
            total_patterns = 0
            for i in range(piano_roll.shape[0] - 2 * window_size):
                pattern1 = piano_roll[i:i+window_size]
                pattern2 = piano_roll[i+window_size:i+2*window_size]
                if np.allclose(pattern1, pattern2):
                    repetitions += 1
                total_patterns += 1
            
            repetition_ratio = 1.0 - min(repetitions / (total_patterns + 1), 1.0)
            reward += 0.2 * repetition_ratio
        
        return float(np.clip(reward, 0.0, 1.0))


class RLHFTrainer:
    """
    REINFORCE-based policy gradient trainer for RLHF.
    """
    
    def __init__(
        self,
        generator_model: nn.Module,
        reward_model: RewardModel,
        learning_rate: float = 1e-5,
        device: torch.device = None
    ):
        """
        Args:
            generator_model: Music generation model (VAE or Transformer)
            reward_model: Trained reward model
            learning_rate: Policy gradient learning rate
            device: Device for training
        """
        self.generator = generator_model
        self.reward_model = reward_model
        self.device = device if device is not None else torch.device("cpu")
        
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.reward_fn = HeuristicRewardFunction()
        

    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        baseline: float = 0.0
    ) -> torch.Tensor:
        """
        Compute REINFORCE loss: -E[reward * log_prob]
        
        Args:
            log_probs: Log probabilities of generated samples
            rewards: Reward for each sample
            baseline: Baseline for variance reduction
        
        Returns:
            Policy loss
        """
        # Advantage = reward - baseline
        advantages = rewards - baseline
        
        # REINFORCE loss
        loss = -torch.mean(log_probs * advantages)
        
        return loss
    
    def train_step(
        self,
        samples_ignored: List[np.ndarray],
        batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Single RLHF training step with REINFORCE.
        
        Args:
            samples_ignored: Ignored (generating fresh samples to maintain gradients)
            batch_size: Batch size for processing
        
        Returns:
            Dict with training metrics
        """
        self.generator.train()
        self.optimizer.zero_grad()
        
        # 1. Generate fresh samples from generator
        # z is the "state" input to the policy
        z = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
        probs = self.generator.decode(z)  # (batch, seq_len, 128)
        
        # 2. Sample an "action" (deterministic thresholding for simplicity, or Bernoulli sampling)
        # We use a differentiable representation for log_prob but fixed action for reward
        with torch.no_grad():
            actions = (probs > 0.5).float()
            
        # 3. Compute rewards for actions using heuristic function
        rewards_list = []
        for i in range(batch_size):
            # Compute reward on the discrete action
            r = self.reward_fn.compute_reward(actions[i].cpu().numpy())
            rewards_list.append(r)
        
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        baseline = rewards.mean().item()
        
        # 4. Compute log probabilities of the chosen actions
        # Using binary cross entropy formula for log_prob of a Bernoulli distribution
        # log P(action|z) = action * log(p) + (1-action) * log(1-p)
        log_probs = actions * torch.log(probs + 1e-9) + (1 - actions) * torch.log(1 - probs + 1e-9)
        log_probs = log_probs.sum(dim=(1, 2))  # Sum over sequence and pitch dimensions
        
        # 5. Compute policy loss: -E[ (reward - baseline) * log_prob ]
        advantages = rewards - baseline
        loss = -torch.mean(log_probs * advantages)
        
        # 6. Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'avg_reward': rewards.mean().item(),
            'baseline': baseline
        }


def simulate_human_survey(
    samples: List[np.ndarray],
    num_participants: int = 10,
    seed: int = 42
) -> Dict:
    """
    Simulate human listening survey by adding noise to heuristic rewards.
    
    Args:
        samples: List of generated samples
        num_participants: Number of simulated participants
        seed: Random seed
    
    Returns:
        Survey data dict
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    reward_fn = HeuristicRewardFunction()
    survey_data = {}
    
    for sample_idx, sample in enumerate(samples):
        # Compute heuristic reward
        base_reward = reward_fn.compute_reward(sample)
        
        # Simulate participant responses with noise
        ratings = []
        for participant_id in range(num_participants):
            # Add participant-specific noise and bias
            noise = np.random.normal(0, 0.15)
            bias = np.random.uniform(-0.1, 0.1)
            
            # Convert to 1-5 scale
            score = base_reward + noise + bias
            score = np.clip(score * 5, 1, 5)  # Scale to [1, 5]
            ratings.append(int(np.round(score)))
        
        survey_data[f"sample_{sample_idx}"] = {
            "ratings": ratings,
            "mean": np.mean(ratings),
            "std": np.std(ratings),
            "base_reward": float(base_reward)
        }
    
    return survey_data


def evaluate_model_with_rewards(
    model: nn.Module,
    samples: List[np.ndarray],
    reward_model: RewardModel,
    device: torch.device = None
) -> Dict:
    """
    Evaluate model on generated samples using reward model.
    
    Args:
        model: Generator model
        samples: List of piano roll samples
        reward_model: Trained reward model
        device: Device
    
    Returns:
        Evaluation metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    reward_model.eval()
    
    with torch.no_grad():
        rewards = []
        for sample in samples:
            # Convert to tensor
            sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get reward
            reward = reward_model(sample_tensor).item()
            rewards.append(reward)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'rewards': rewards
    }


if __name__ == "__main__":
    print("RLHF module loaded.")
    
    # Test reward function
    reward_fn = HeuristicRewardFunction()
    dummy_sample = np.random.randn(256, 128)
    dummy_sample = np.clip(dummy_sample, 0, 1)
    reward = reward_fn.compute_reward(dummy_sample)
    print(f"Dummy sample reward: {reward:.4f}")
