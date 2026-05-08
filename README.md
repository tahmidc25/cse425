# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425/EEE474 Neural Networks — Spring 2026

## Project Overview
This project implements a comprehensive pipeline for music generation using unsupervised deep learning techniques. It covers four progressive tasks: LSTM Autoencoder (single-genre), Variational Autoencoder (multi-genre), Transformer (long-form generation), and RLHF (human preference tuning).

## Dataset
The project uses three public datasets:

| Dataset | Genre | Link |
|---------|-------|------|
| MAESTRO v2.0.0 | Classical Piano | [Download](https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip) |
| Lakh MIDI Dataset | Multi-Genre Collection | [Download](http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz) |
| Groove MIDI Dataset | Jazz / Drums / Rhythm | [Download](https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip) |

**Google Drive Links:**
- **Generated MIDI Samples:** [MIDI Folder](https://drive.google.com/drive/folders/1uzn5TuCNnX0NhhR_5V-xmwnIRJ9U6bV9?usp=drive_link)
- **Final Project Report (PDF):** [Project Report](https://drive.google.com/file/d/1MrLcGmdKXUFBFOduWB7VrFtVaoAfQDGq/view?usp=drive_link)

## Project Structure

```
music-generation-unsupervised/
├── README.md                          # Project documentation (this file)
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw_midi/                      # Raw downloaded MIDI files
│   ├── processed/                     # Preprocessed piano rolls and tokens
│   └── train_test_split/              # Train/validation/test splits
├── notebooks/
│   ├── preprocessing.ipynb            # Data preprocessing walkthrough
│   └── baseline_markov.ipynb          # Baseline Markov chain experiments
├── src/
│   ├── config.py                      # Global configuration and hyperparameters
│   ├── preprocessing/
│   │   ├── midi_parser.py             # Main dataset preprocessing orchestrator
│   │   ├── tokenizer.py              # MIDI-to-token conversion (228-token vocabulary)
│   │   └── piano_roll.py             # MIDI-to-piano-roll conversion (128 pitches × T steps)
│   ├── models/
│   │   ├── autoencoder.py            # Task 1: LSTM Autoencoder (Encoder + Decoder + MSE loss)
│   │   ├── vae.py                    # Task 2: Variational Autoencoder (KL-divergence + reparameterization)
│   │   ├── transformer.py           # Task 3: GPT-style Transformer (causal masking, 8 heads, 6 layers)
│   │   ├── diffusion.py             # Diffusion model placeholder (optional extension)
│   │   ├── rlhf.py                  # Task 4: RLHF Reward Model + REINFORCE policy gradient trainer
│   │   └── baselines.py            # Baseline models: Random Note Generator + Markov Chain
│   ├── training/
│   │   ├── train_ae.py              # Training script for Task 1 (LSTM Autoencoder)
│   │   ├── train_vae.py             # Training script for Task 2 (VAE with β-KL)
│   │   └── train_transformer.py     # Training script for Task 3 (Transformer)
│   ├── evaluation/
│   │   ├── metrics.py               # All evaluation metrics (pitch histogram, rhythm diversity, etc.)
│   │   ├── pitch_histogram.py       # Pitch histogram similarity (L1 distance, 12-bin chromatic)
│   │   └── rhythm_score.py          # Rhythm diversity score (unique durations / total notes)
│   └── generation/
│       ├── sample_latent.py         # Latent space sampling utilities
│       ├── generate_music.py        # Main MIDI generation from trained models
│       └── midi_export.py           # MIDI file export utilities
├── outputs/
│   ├── generated_midis/              # Generated MIDI files organized by task
│   │   ├── task1/                    # 5 samples from LSTM Autoencoder
│   │   ├── task2/                    # 8 samples + 5 interpolations from VAE
│   │   ├── task3/                    # 10 long-sequence samples from Transformer
│   │   └── task4/                    # 10 RLHF-tuned samples + baselines
│   ├── plots/                        # Loss curves, metric comparisons
│   └── survey_results/               # Human listening survey data and charts
└── report/
    ├── final_report.tex              # IEEE-format LaTeX report (Abstract → Conclusion)
    ├── architecture_diagrams/        # Model architecture diagrams
    └── references.bib                # Bibliography (6 citations)
```

## File Descriptions

### Preprocessing (`src/preprocessing/`)
| File | Description |
|------|-------------|
| `midi_parser.py` | Main orchestrator that processes MIDI datasets (MAESTRO, Lakh, Groove) into piano rolls and tokens |
| `piano_roll.py` | Converts MIDI to piano roll representation: `(seq_len, 128)` array with 16 steps/bar resolution |
| `tokenizer.py` | Converts MIDI to token sequences using a 228-token vocabulary (NOTE_ON, VELOCITY, DURATION, etc.) |

### Models (`src/models/`)
| File | Description |
|------|-------------|
| `autoencoder.py` | **Task 1** — LSTM Autoencoder. Encoder: 2-layer LSTM (hidden=256) → latent_dim=64. Decoder reconstructs piano roll. Loss: MSE |
| `vae.py` | **Task 2** — Variational Autoencoder. Outputs μ and log(σ²), uses reparameterization trick. Loss: Reconstruction + β·KL divergence (β=0.5) |
| `transformer.py` | **Task 3** — GPT-style Transformer. d_model=512, nhead=8, 6 layers, causal masking, max_seq=512 tokens. Loss: Cross-entropy |
| `rlhf.py` | **Task 4** — RLHF module. LSTM-based Reward Model + Heuristic Reward Function + REINFORCE policy gradient trainer |
| `baselines.py` | Two baseline models: Random Note Generator (uniform pitch/duration) and Markov Chain (order-1 pitch transitions) |
| `diffusion.py` | Placeholder for optional Diffusion model extension |

### Training (`src/training/`)
| File | Description |
|------|-------------|
| `train_ae.py` | Trains Task 1 LSTM Autoencoder. 50 epochs, Adam (lr=1e-3), batch_size=64 |
| `train_vae.py` | Trains Task 2 VAE on multi-genre data. 50 epochs, Adam (lr=1e-3), batch_size=32 |
| `train_transformer.py` | Trains Task 3 Transformer. 30 epochs, AdamW (lr=1e-4), batch_size=16 |

### Evaluation (`src/evaluation/`)
| File | Description |
|------|-------------|
| `metrics.py` | Complete evaluation suite: Pitch Histogram Similarity, Rhythm Diversity, Repetition Ratio, Perplexity, Human Listening Score |
| `pitch_histogram.py` | L1 distance on 12-bin chromatic pitch distribution |
| `rhythm_score.py` | Rhythm diversity = unique durations / total notes |

### Generation (`src/generation/`)
| File | Description |
|------|-------------|
| `generate_music.py` | Generates MIDI files from trained model checkpoints (AE, VAE, Transformer) |
| `sample_latent.py` | Utilities for sampling and interpolating in latent space |
| `midi_export.py` | MIDI file export utilities |

## Installation

```bash
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Preprocessing
```bash
python src/preprocessing/midi_parser.py
```

### 2. Train Models
```bash
python src/training/train_ae.py            # Task 1: LSTM Autoencoder
python src/training/train_vae.py           # Task 2: VAE
python src/training/train_transformer.py   # Task 3: Transformer
```

### 3. Generate Music
```bash
python src/generation/generate_music.py --model ae --checkpoint outputs/checkpoints/...
python src/generation/generate_music.py --model vae --checkpoint outputs/checkpoints/...
python src/generation/generate_music.py --model transformer --checkpoint outputs/checkpoints/...
```

### 4. Evaluate
```bash
python src/evaluation/metrics.py
```

## Model Architectures

### Task 1: LSTM Autoencoder
- **Encoder**: 2-layer LSTM (hidden_dim=256) → latent_dim=64
- **Decoder**: latent_dim=64 → 2-layer LSTM → 128 piano notes
- **Loss**: MSE reconstruction loss `L_AE = ||X - X̂||²`
- **Training**: 50 epochs, Adam (lr=1e-3), batch_size=64

### Task 2: Variational Autoencoder (Multi-Genre)
- **Encoder**: 2-layer LSTM → μ and log(σ²) outputs
- **Latent Dimension**: 32
- **Reparameterization**: z = μ + σ ⊙ ε, ε ~ N(0, I)
- **Loss**: L_VAE = L_recon + β·D_KL(q||p), β=0.5
- **Features**: Latent interpolation, genre blending

### Task 3: Transformer (Long Sequences)
- **Architecture**: Decoder-only GPT-style, causal masking
- **Specs**: d_model=512, nhead=8, num_layers=6, d_ff=2048
- **Max Seq Length**: 512 tokens
- **Loss**: Cross-entropy (autoregressive)
- **Metric**: Perplexity = exp(1/T · L_TR)

### Task 4: RLHF (Human Preference Tuning)
- **Reward Model**: 2-layer LSTM scoring head → [0, 1]
- **Heuristic Scoring**: Pitch variety, rhythm density, scale conformity
- **Training**: REINFORCE policy gradient, 300 steps
- **Human Survey**: 10 simulated participants, 1-5 scale

## Baselines
1. **Random Note Generator**: Random pitch ∈ [40, 80], random duration, random velocity
2. **Markov Chain**: Order-1 pitch transition probabilities learned from data

## Evaluation Metrics
| Metric | Formula | Description |
|--------|---------|-------------|
| Pitch Histogram Similarity | H(p,q) = Σ|pᵢ - qᵢ| | L1 distance on 12-bin chromatic distribution |
| Rhythm Diversity | D = unique_durations / total_notes | Higher = more rhythmic variety |
| Repetition Ratio | R = repeated_patterns / total_patterns | Lower = less repetitive |
| Perplexity | exp(1/T · L_TR) | Lower = better sequence modeling |
| Human Listening Score | Score ∈ [1, 5] | Simulated survey (10 participants) |

## Results Summary

| Model | Loss | Perplexity | Rhythm Div. | Human Score | Genre Control |
|-------|------|-----------|-------------|-------------|---------------|
| Random Generator | — | — | 0.48 | 2.1 | None |
| Markov Chain | — | 156 | 0.52 | 2.4 | Weak |
| Task 1: AE | 0.0234 | — | 0.62 | 2.6 | Single Genre |
| Task 2: VAE | 0.0198 | — | 0.68 | 2.8 | Moderate |
| Task 3: Transformer | 0.0345 | 28.4 | 0.74 | 3.1 | Strong |
| Task 4: RLHF | 0.0312 | 32.1 | 0.76 | 3.3 | Strongest |

## Reproducibility
- Random seed: 42
- GPU: CUDA if available, CPU fallback
- Checkpoint saving: Every 10 epochs
- Dependencies: See `requirements.txt`

## Authors
CSE425/EEE474 Neural Networks — Spring 2026 Course Project

| Name | Role |
|------|------|
| Asif Ahmed Joy | Task 1 & Task 2 |
| Tahmid Chowdhury | Task 3 & Task 4 |

## Group Member Contributions

### Asif Ahmed Joy
| Area | Contribution |
|------|-------------|
| **Task 1: LSTM Autoencoder** | Designed and implemented the LSTM Encoder-Decoder architecture (`autoencoder.py`), wrote the MSE reconstruction loss function, trained the model for 50 epochs, and generated 5 MIDI samples |
| **Task 2: VAE Multi-Genre** | Extended the autoencoder into a Variational Autoencoder (`vae.py`), implemented the reparameterization trick (z = μ + σ ⊙ ε), KL-divergence loss with β=0.5, latent space interpolation experiment, and generated 8 multi-genre MIDI samples |
| **Preprocessing** | Built the MIDI-to-piano-roll conversion pipeline (`piano_roll.py`, `midi_parser.py`), normalized timing to 16 steps/bar |
| **Baselines** | Implemented the Random Note Generator and Markov Chain baseline models (`baselines.py`) |
| **Report** | Wrote the Methodology sections for Task 1 and Task 2, dataset description, and preprocessing pipeline |

### Tahmid Chowdhury
| Area | Contribution |
|------|-------------|
| **Task 3: Transformer** | Designed and implemented the GPT-style Transformer architecture (`transformer.py`) with causal masking, 8 attention heads, 6 layers, d_model=512. Trained with cross-entropy loss and evaluated using perplexity. Generated 10 long-sequence MIDI compositions |
| **Task 4: RLHF** | Implemented the LSTM-based Reward Model and Heuristic Reward Function (`rlhf.py`), built the REINFORCE policy gradient trainer, ran 300 RL training steps, conducted simulated human listening survey (10 participants), and generated 10 RLHF-tuned MIDI samples with before/after comparison |
| **Tokenization** | Built the MIDI-to-token conversion pipeline (`tokenizer.py`) with 228-token vocabulary for Transformer training |
| **Evaluation** | Implemented all evaluation metrics (`metrics.py`): Pitch Histogram Similarity, Rhythm Diversity, Repetition Ratio, Perplexity, Human Listening Score |
| **Report** | Wrote the Methodology sections for Task 3 and Task 4, results analysis, comparison table, and conclusion |

## References
See `report/references.bib` for complete bibliography.
