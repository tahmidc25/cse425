"""
MIDI generation utilities - convert models outputs to playable MIDI files.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import pianoroll_to_midi, tokens_to_midi
from models import LSTMAutoencoder, VariationalAutoencoder, MusicTransformer


def generate_from_latent_space(
    model_path: str,
    model_type: str,
    num_samples: int = 10,
    output_dir: str = "outputs/generated_midis"
):
    """
    Generate MIDI samples by sampling from latent space.
    
    Args:
        model_path: Path to trained model checkpoint
        model_type: "ae" (Task 1), "vae" (Task 2), or "transformer" (Task 3)
        num_samples: Number of samples to generate
        output_dir: Output directory
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    if model_type == "ae":
        model = LSTMAutoencoder(input_dim=128, hidden_dim=256, latent_dim=64, seq_len=256)
    elif model_type == "vae":
        model = VariationalAutoencoder(input_dim=128, hidden_dim=256, latent_dim=32, seq_len=256)
    elif model_type == "transformer":
        model = MusicTransformer(vocab_size=228, d_model=512, nhead=8, num_layers=6)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate
    with torch.no_grad():
        for i in range(num_samples):
            if model_type == "transformer":
                # Autoregressive generation
                tokens = model.generate(
                    start_token=0,
                    max_length=512,
                    temperature=0.9,
                    device=device
                )
                tokens_np = tokens.cpu().numpy()[0]
                midi = tokens_to_midi(tokens_np, fs=16, output_path=None)
            else:
                # Latent space sampling
                z = torch.randn(1, model.latent_dim if hasattr(model, 'latent_dim') else 32, device=device)
                piano_roll = model.decode(z)
                piano_roll = torch.sigmoid(piano_roll).cpu().numpy()[0]
                midi = pianoroll_to_midi(piano_roll, fs=16, output_path=None)
            
            # Save MIDI
            output_path = Path(output_dir) / f"generated_{i:02d}.mid"
            midi.write(str(output_path))
            print(f"Generated {output_path}")


def interpolate_latent_space(
    model_path: str,
    num_steps: int = 5,
    output_dir: str = "outputs/generated_midis/interpolation"
):
    """
    Interpolate between two points in latent space.
    
    Args:
        model_path: Path to trained VAE checkpoint
        num_steps: Number of interpolation steps
        output_dir: Output directory
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = VariationalAutoencoder(
        input_dim=128,
        hidden_dim=256,
        latent_dim=32,
        seq_len=256
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Sample two points
    with torch.no_grad():
        z1 = torch.randn(1, 32, device=device)
        z2 = torch.randn(1, 32, device=device)
        
        for i, alpha in enumerate(np.linspace(0, 1, num_steps)):
            # Interpolate
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Decode
            piano_roll = model.decode(z_interp)
            piano_roll = torch.sigmoid(piano_roll).cpu().numpy()[0]
            
            # Save MIDI
            output_path = Path(output_dir) / f"interp_{i:02d}.mid"
            midi = pianoroll_to_midi(piano_roll, fs=16, output_path=str(output_path))
            print(f"Interpolated step {i}/{num_steps-1}: {output_path}")


def batch_generate_all_tasks(
    checkpoint_dir: str = "outputs/checkpoints",
    output_dir: str = "outputs/generated_midis"
):
    """
    Generate samples from all trained tasks.
    
    Args:
        checkpoint_dir: Directory with checkpoints
        output_dir: Output directory
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Task 1
    task1_checkpoints = list(checkpoint_dir.glob("task1_checkpoint_*.pt"))
    if task1_checkpoints:
        latest = max(task1_checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"Generating from Task 1: {latest}")
        generate_from_latent_space(str(latest), "ae", num_samples=5, output_dir=f"{output_dir}/task1")
    
    # Task 2
    task2_checkpoints = list(checkpoint_dir.glob("task2_checkpoint_*.pt"))
    if task2_checkpoints:
        latest = max(task2_checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"Generating from Task 2: {latest}")
        generate_from_latent_space(str(latest), "vae", num_samples=8, output_dir=f"{output_dir}/task2")
        # Also interpolate
        print("Generating interpolations...")
        interpolate_latent_space(str(latest), num_steps=5, output_dir=f"{output_dir}/task2/interp")
    
    # Task 3
    task3_checkpoints = list(checkpoint_dir.glob("task3_checkpoint_*.pt"))
    if task3_checkpoints:
        latest = max(task3_checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"Generating from Task 3: {latest}")
        generate_from_latent_space(str(latest), "transformer", num_samples=10, output_dir=f"{output_dir}/task3")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MIDI from trained models")
    parser.add_argument("--model", choices=["ae", "vae", "transformer"], help="Model type")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--output-dir", default="outputs/generated_midis", help="Output directory")
    parser.add_argument("--interpolate", action="store_true", help="For VAE: interpolate instead")
    
    args = parser.parse_args()
    
    if args.interpolate:
        interpolate_latent_space(args.checkpoint, num_steps=args.num_samples, output_dir=args.output_dir)
    else:
        generate_from_latent_space(
            args.checkpoint,
            args.model,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
