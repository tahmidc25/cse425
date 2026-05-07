# Unsupervised Neural Network for Multi-Genre Music Generation

## Project Overview
This project implements a comprehensive pipeline for music generation using unsupervised deep learning techniques. It covers four progressive tasks: LSTM Autoencoder (single-genre), Variational Autoencoder (multi-genre), Transformer (long-form generation), and RLHF (human preference tuning).

## Dataset
The project uses three public datasets:
https://drive.google.com/drive/folders/1qrmyBY9Sb2w9phDAgrh2LdIAuwWf4VQz?usp=drive_link 
1. **MAESTRO v2.0.0** - Classical piano music (https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip)
2. **Lakh MIDI Dataset** - Multi-genre MIDI (http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz)
3. **Groove MIDI Dataset** - Jazz, drums, rhythm (https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip)

## Project Structure
```
music-generation-unsupervised/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_midi/          # Raw downloaded MIDI files
│   ├── processed/         # Preprocessed piano rolls and tokens
│   └── train_test_split/  # Train/validation/test splits
├── src/
│   ├── preprocessing/     # Data preprocessing pipeline
│   ├── models/            # Task-specific model architectures
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation metrics
│   └── generation/        # MIDI generation utilities
├── outputs/
│   ├── generated_midis/   # Generated MIDI files (organized by task)
│   ├── plots/             # Training curves, latent space visualizations
│   └── results/           # Evaluation results (CSV, JSON)
└── report/
    ├── final_report.tex   # LaTeX project report
    └── references.bib     # Bibliography
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Download Datasets
```bash
cd data/raw_midi
# Download MAESTRO
wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip
unzip maestro-v2.0.0-midi.zip

# Download Lakh MIDI (may require mirroring - see preprocessing docs)
# Download Groove
wget https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip
unzip groove-v1.0.0.zip
```

### 2. Preprocessing
```bash
python src/preprocessing/dataset.py
```
This will:
- Convert MIDI files to piano rolls (16 steps/bar resolution)
- Tokenize sequences for transformer training
- Create train/test splits

### 3. Train Models

**Task 1: LSTM Autoencoder**
```bash
python src/training/train_task1.py
```

**Task 2: Variational Autoencoder**
```bash
python src/training/train_task2.py
```

**Task 3: Transformer**
```bash
python src/training/train_task3.py
```

**Task 4: RLHF**
```bash
python src/training/train_task4.py
```

### 4. Generate Music
```bash
python src/generation/generate_midi.py --task 1 --num-samples 5
python src/generation/generate_midi.py --task 2 --num-samples 8 --interpolate
python src/generation/generate_midi.py --task 3 --num-samples 10
python src/generation/generate_midi.py --task 4 --num-samples 10
```

### 5. Evaluate
```bash
python src/evaluation/metrics.py
```

## Model Architectures

### Task 1: LSTM Autoencoder
- **Encoder**: 2-layer LSTM (hidden_dim=256) → latent_dim=64
- **Decoder**: latent_dim=64 → 2-layer LSTM → 128 piano notes
- **Loss**: MSE reconstruction loss
- **Training**: 50+ epochs, Adam (lr=1e-3), batch_size=64

### Task 2: Variational Autoencoder (Multi-Genre)
- **Encoder**: 2-layer LSTM with μ and σ outputs
- **Latent Dimension**: 32
- **Loss**: Reconstruction + β·KL divergence (β=0.5)
- **Training**: 50+ epochs, multi-genre dataset
- **Features**: Latent interpolation, genre conditioning

### Task 3: Transformer (Long Sequences)
- **Architecture**: Decoder-only GPT-style, causal masking
- **d_model=512**: nhead=8, num_layers=6, feedforward_dim=2048
- **Max Seq Length**: 512 tokens
- **Loss**: Cross-entropy with autoregressive sampling
- **Evaluation**: Perplexity computation

### Task 4: RLHF (Human Preference Tuning)
- **Base Model**: VAE or Transformer generator
- **Reward Model**: LSTM-based, trained on heuristic scoring
- **Scoring**: Pitch variety, rhythm density, scale conformity
- **Training**: REINFORCE policy gradient, 300+ steps
- **Comparison**: Before/after human survey simulation

## Key Results

All results are saved in `outputs/`:
- **Generated MIDI**: Organized by task (task1/ through task4/, baselines/)
- **Plots**: Loss curves, latent space visualizations, survey results
- **Metrics**: Comprehensive comparison table (comparison_table.csv)

## Baselines

1. **Random Note Generator**: Random pitch, duration, velocity
2. **Markov Chain**: Order-1 or Order-2 pitch transition model

## Evaluation Metrics

1. **Pitch Histogram Similarity**: L1 distance on 12-bin chromatic pitch distribution
2. **Rhythm Diversity**: Ratio of unique note durations
3. **Repetition Ratio**: Percentage of repeated 4-note patterns
4. **Perplexity**: Transformer sequence likelihood metric
5. **Human Listening Score**: Simulated survey (1-5 scale)

## Report

The final LaTeX report (`report/final_report.tex`) includes:
- Dataset and preprocessing details
- Detailed mathematical formulations for all tasks
- Experimental setup and hyperparameters
- Comprehensive results and comparisons
- Analysis of latent interpolation and RLHF improvements

## Reproducibility

All experiments use:
- Random seed: 42
- GPU: CUDA if available, CPU fallback
- Checkpoint saving: Every 10 epochs
- Version control: See dependencies in requirements.txt

## Authors
CSE425/EEE474 Neural Networks - Spring 2026 Course Project

## References
See `report/references.bib` for complete bibliography.
