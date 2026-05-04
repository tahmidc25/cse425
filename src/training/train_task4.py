"""
Training script for Task 4: RLHF (Reinforcement Learning from Human Feedback)
Trains a reward model and fine-tunes generator using policy gradients.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.task2_vae import VariationalAutoencoder
from models.task4_rlhf import (
    RewardModel,
    RLHFTrainer,
    HeuristicRewardFunction,
    simulate_human_survey,
    evaluate_model_with_rewards
)
from preprocessing import pianoroll_to_midi


class RewardModelTrainer:
    """Trainer for the reward model."""
    
    def __init__(
        self,
        reward_model: RewardModel,
        device: torch.device,
        learning_rate: float = 1e-3
    ):
        """
        Args:
            reward_model: Reward model to train
            device: Device
            learning_rate: Learning rate
        """
        self.reward_model = reward_model.to(device)
        self.device = device
        self.optimizer = optim.Adam(reward_model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Average loss
        """
        self.reward_model.train()
        total_loss = 0.0
        num_batches = 0
        
        reward_fn = HeuristicRewardFunction()
        
        for batch_pr in train_loader:
            # Unpack tuple from DataLoader
            if isinstance(batch_pr, (tuple, list)):
                batch_pr = batch_pr[0]
            batch_pr = batch_pr.to(self.device)
            batch_size = batch_pr.size(0)
            
            # Compute heuristic rewards
            heuristic_rewards = []
            for i in range(batch_size):
                piano_roll = batch_pr[i].cpu().numpy()
                reward = reward_fn.compute_reward(piano_roll)
                heuristic_rewards.append(reward)
            
            target_rewards = torch.tensor(heuristic_rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            
            # Forward pass
            predicted_rewards = self.reward_model(batch_pr)
            
            # Loss
            loss = self.criterion(predicted_rewards, target_rewards)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, train_loader: DataLoader, num_epochs: int = 10):
        """Train reward model."""
        for epoch in range(num_epochs):
            loss = self.train_epoch(train_loader)
            if (epoch + 1) % 5 == 0:
                print(f"Reward Model - Epoch {epoch+1}/{num_epochs}: Loss={loss:.4f}")


class Task4RLHFTrainer:
    """Main RLHF trainer combining reward model and policy gradient."""
    
    def __init__(
        self,
        generator: VariationalAutoencoder,
        device: torch.device,
        learning_rate: float = 1e-5,
        output_dir: str = "outputs"
    ):
        """
        Args:
            generator: Pre-trained VAE generator
            device: Device
            learning_rate: Policy gradient learning rate
            output_dir: Output directory
        """
        self.generator = generator.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create reward model
        self.reward_model = RewardModel(input_dim=128, hidden_dim=256).to(device)
        
        # RLHF trainer
        self.rlhf_trainer = RLHFTrainer(
            generator,
            self.reward_model,
            learning_rate=learning_rate,
            device=device
        )
        
        self.training_rewards = []
        self.baseline_rewards = []
    
    def train_reward_model(self, train_loader: DataLoader, num_epochs: int = 10):
        """Train reward model on heuristic labels."""
        print("Training reward model...")
        reward_trainer = RewardModelTrainer(self.reward_model, self.device)
        reward_trainer.train(train_loader, num_epochs=num_epochs)
    
    def generate_samples(self, num_samples: int = 10) -> list:
        """Generate samples from current generator."""
        self.generator.eval()
        samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                z = torch.randn(1, self.generator.latent_dim, device=self.device)
                piano_roll = self.generator.decode(z)
                piano_roll = torch.sigmoid(piano_roll)  # Ensure [0, 1]
                samples.append(piano_roll.cpu().numpy()[0])
        
        return samples
    
    def train_with_rl(self, num_steps: int = 300, batch_size: int = 4):
        """
        Train generator with RLHF.
        
        Args:
            num_steps: Number of RL training steps
            batch_size: Batch size for generation
        """
        print("Starting RLHF training...")
        
        for step in range(num_steps):
            # Generate samples
            samples = self.generate_samples(num_samples=batch_size)
            
            # Train step
            metrics = self.rlhf_trainer.train_step(samples, batch_size=batch_size)
            
            self.training_rewards.append(metrics['avg_reward'])
            
            if (step + 1) % 50 == 0:
                print(f"RLHF Step {step+1}/{num_steps}: Reward={metrics['avg_reward']:.4f}")
    
    def evaluate_before_after(self, num_samples: int = 10):
        """
        Generate samples before and after RLHF and compare rewards.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Dict with comparison stats
        """
        samples = self.generate_samples(num_samples=num_samples)
        
        # Get rewards
        self.reward_model.eval()
        with torch.no_grad():
            rewards = []
            for sample in samples:
                sample_tensor = torch.tensor(sample, dtype=torch.float32, device=self.device).unsqueeze(0)
                reward = self.reward_model(sample_tensor).item()
                rewards.append(reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'rewards': rewards
        }
    
    def plot_rl_training(self, output_path: str = "outputs/plots/survey_results/rlhf_comparison.png"):
        """Plot RLHF training progression."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_rewards, label='Training Rewards', marker='o', markersize=4)
        plt.axhline(y=np.mean(self.training_rewards[:10]), color='r', linestyle='--', label='Initial Average')
        plt.axhline(y=np.mean(self.training_rewards[-10:]), color='g', linestyle='--', label='Final Average')
        plt.xlabel('RLHF Step')
        plt.ylabel('Average Reward')
        plt.title('Task 4: RLHF - Training Progression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
        plt.close()
    
    def save_samples_and_survey(self, num_samples: int = 10):
        """Generate samples and simulate survey."""
        # Generate samples
        samples = self.generate_samples(num_samples=num_samples)
        
        # Save as MIDI
        output_dir = self.output_dir / "generated_midis" / "task4"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, sample in enumerate(samples):
            output_path = output_dir / f"sample_{i:02d}.mid"
            pianoroll_to_midi(sample, fs=16, output_path=str(output_path))
            print(f"Saved {output_path}")
        
        # Simulate human survey
        survey_data = simulate_human_survey(samples, num_participants=10)
        
        # Save survey data
        survey_path = self.output_dir / "results" / "survey_data.json"
        survey_path.parent.mkdir(parents=True, exist_ok=True)
        with open(survey_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            survey_serializable = {}
            for key, val in survey_data.items():
                survey_serializable[key] = {
                    'ratings': [int(x) for x in val['ratings']],
                    'mean': float(val['mean']),
                    'std': float(val['std']),
                    'base_reward': float(val['base_reward'])
                }
            json.dump(survey_serializable, f, indent=2)
        
        print(f"Saved survey data to {survey_path}")
        
        return survey_data


def create_dummy_dataset(num_samples: int = 50, seq_len: int = 256):
    """Create dummy dataset."""
    piano_rolls = np.random.rand(num_samples, seq_len, 128).astype(np.float32)
    tensor = torch.tensor(piano_rolls)
    
    train_size = int(0.8 * num_samples)
    train_tensor = tensor[:train_size]
    
    return train_tensor


def main():
    """Main RLHF training script."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    RL_STEPS = 300
    
    print(f"Using device: {DEVICE}")
    
    # Create pre-trained generator (VAE)
    generator = VariationalAutoencoder(
        input_dim=128,
        hidden_dim=256,
        latent_dim=32,
        seq_len=256,
        beta=0.5
    )
    
    # Create RLHF trainer
    rlhf_trainer = Task4RLHFTrainer(generator, DEVICE, learning_rate=1e-5)
    
    # Prepare training data for reward model
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.dataset_loader import create_real_dataset
    print("Creating real dataset...")
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_dataset, _ = create_real_dataset(str(data_dir), batch_size=BATCH_SIZE, seq_len=256, is_tokens=False, max_samples=1000)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Train reward model
    rlhf_trainer.train_reward_model(train_loader, num_epochs=10)
    
    # Save baseline samples (before RLHF)
    print("Generating baseline samples...")
    baseline_samples = rlhf_trainer.generate_samples(num_samples=10)
    baseline_dir = Path("outputs/generated_midis/task4/baseline")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(baseline_samples):
        output_path = baseline_dir / f"baseline_{i:02d}.mid"
        pianoroll_to_midi(sample, fs=16, output_path=str(output_path))
    
    baseline_eval = rlhf_trainer.evaluate_before_after(num_samples=10)
    print(f"Baseline average reward: {baseline_eval['mean_reward']:.4f}")
    
    # RLHF training
    print("Starting RLHF training...")
    rlhf_trainer.train_with_rl(num_steps=RL_STEPS, batch_size=BATCH_SIZE)
    
    # Evaluate after RLHF
    print("Evaluating after RLHF...")
    after_eval = rlhf_trainer.evaluate_before_after(num_samples=10)
    print(f"After RLHF average reward: {after_eval['mean_reward']:.4f}")
    
    # Save final samples and survey
    rlhf_trainer.save_samples_and_survey(num_samples=10)
    
    # Plot training progression
    rlhf_trainer.plot_rl_training()
    
    # Save comparison
    comparison = {
        'baseline': {
            'mean_reward': float(baseline_eval['mean_reward']),
            'std_reward': float(baseline_eval['std_reward'])
        },
        'after_rlhf': {
            'mean_reward': float(after_eval['mean_reward']),
            'std_reward': float(after_eval['std_reward'])
        },
        'improvement': float(after_eval['mean_reward'] - baseline_eval['mean_reward'])
    }
    
    comparison_path = Path("outputs/results/rlhf_comparison.json")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Saved comparison to {comparison_path}")
    print("Task 4 RLHF training complete!")


if __name__ == "__main__":
    main()
