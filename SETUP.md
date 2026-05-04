# Setup Guide - Music Generation Project

## System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.8+
- **GPU**: NVIDIA GPU with CUDA support (highly recommended; CPU is 100x slower)
- **Storage**: ~100 GB for datasets + models + outputs
- **RAM**: 16 GB minimum, 32 GB recommended

## Installation

### 1. Clone or Extract Project

```bash
cd /path/to/music-generation-unsupervised
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Using conda
conda create -n music-gen python=3.10
conda activate music-gen
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support (CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dataset Download

### Automatic Download (Recommended)

```bash
# From project root
python scripts/download_datasets.py
```

### Manual Download

```bash
# 1. MAESTRO v2.0.0
cd data/raw_midi/maestro
wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip
unzip maestro-v2.0.0-midi.zip
cd ../..

# 2. Lakh MIDI Dataset
cd lakh_midi
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xzf lmd_full.tar.gz
cd ../..

# 3. Groove MIDI Dataset
cd groove
wget https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip
unzip groove-v1.0.0.zip
cd ../..
```

**Note**: Lakh MIDI is large (~23 GB). Consider downloading a subset for faster processing.

## Quick Start

### Option 1: Interactive Menu (Recommended)

```bash
python main.py
```

Select options from the interactive menu.

### Option 2: Command Line

```bash
# Full pipeline (preprocessing + training + evaluation)
python main.py --full

# Run specific task
python main.py --task 1
python main.py --task 2
python main.py --task 3
python main.py --task 4

# Evaluate only
python main.py --eval-only

# Verify outputs
python main.py --verify

# Create submission archive
python main.py --archive
```

### Option 3: Manual Step-by-Step

```bash
# 1. Preprocess data
python src/preprocessing/dataset.py

# 2. Train Task 1
python src/training/train_task1.py

# 3. Train Task 2
python src/training/train_task2.py

# 4. Train Task 3
python src/training/train_task3.py

# 5. Train Task 4
python src/training/train_task4.py

# 6. Evaluate
python src/evaluation/run_evaluation.py
```

## Expected Training Times (on NVIDIA P100)

| Task | Phase | Time |
|------|-------|------|
| 1 | Preprocessing | ~30 min (full dataset) |
| 1 | Training | ~4 hours (50 epochs) |
| 2 | Training | ~5 hours (50 epochs, multi-genre) |
| 3 | Training | ~8 hours (30 epochs, large model) |
| 4 | Training | ~6 hours (300 RL steps) |
| **Total** | - | **~23 hours** |

**Notes**:
- First run downloads/processes data
- Subsequent runs can skip preprocessing
- Reduce epochs/batch size for faster iteration on CPU
- GPU training 50-100x faster than CPU

## Project Structure

```
music-generation-unsupervised/
├── README.md              # Project overview
├── SETUP.md               # This file
├── requirements.txt       # Python dependencies
├── main.py                # Pipeline orchestrator
├── data/
│   ├── raw_midi/          # Downloaded MIDI files
│   ├── processed/         # Preprocessed piano rolls & tokens
│   └── train_test_split/  # Data split JSON files
├── src/
│   ├── preprocessing/     # MIDI conversion pipeline
│   ├── models/            # Task-specific architectures
│   ├── training/          # Training scripts (4 tasks)
│   ├── evaluation/        # Metrics & evaluation
│   └── generation/        # MIDI generation utilities
├── outputs/
│   ├── generated_midis/   # Generated MIDI files
│   │   ├── task1/         # 5 samples
│   │   ├── task2/         # 8 samples + interpolation
│   │   ├── task3/         # 10 samples
│   │   ├── task4/         # 10 samples
│   │   └── baselines/     # Random + Markov
│   ├── plots/             # Visualizations
│   │   ├── loss_curves/
│   │   ├── metric_comparison/
│   │   ├── latent_space/
│   │   └── survey_results/
│   └── results/           # CSV & JSON results
└── report/
    ├── final_report.tex   # LaTeX paper
    ├── references.bib     # Bibliography
    └── architecture_diagrams/
```

## Configuration

### Default Hyperparameters

Edit training scripts to customize:

**src/training/train_task1.py**:
```python
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
```

**src/training/train_task2.py**:
```python
BETA = 0.5  # KL weight
LEARNING_RATE = 1e-3
```

**src/training/train_task3.py**:
```python
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6
```

**src/training/train_task4.py**:
```python
RL_STEPS = 300
LEARNING_RATE = 1e-5  # Policy gradient LR
```

### Data Configuration

Modify `src/preprocessing/dataset.py`:
```python
fs = 16              # Steps per beat
piano_roll_len = 256 # Sequence length
token_seq_len = 512  # Token sequence length
```

## Troubleshooting

### GPU Memory Error

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size in training scripts:
```python
BATCH_SIZE = 16  # was 64
```

### Dataset Download Fails

**MAESTRO/Groove**: Try alternative mirrors or download via web browser manually.

**Lakh MIDI**: Very large (~23 GB). Consider subset:
```bash
# Download only classical subset (~2 GB)
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz
```

### MIDI Processing Errors

Some MIDI files may be malformed. The pipeline logs errors and continues.

Check logs:
```bash
python src/preprocessing/dataset.py 2>&1 | tee preprocessing.log
```

### Out of Disk Space

Check space:
```bash
df -h
```

If needed, reduce:
- Checkpoint frequency (save every 50 instead of 10 epochs)
- Number of generated samples

## Evaluation & Output

After training, evaluate:
```bash
python src/evaluation/run_evaluation.py
```

Outputs:
- `outputs/results/comparison_table.csv` - Model metrics
- `outputs/plots/metric_comparison/` - Comparison plots
- `outputs/results/survey_data.json` - Human survey simulation
- `outputs/generated_midis/` - Generated MIDI files

### Play MIDI Files

```bash
# Linux
timidity outputs/generated_midis/task1/sample_00.mid

# macOS
open -a GarageBand outputs/generated_midis/task1/sample_00.mid

# Windows
start outputs/generated_midis/task1/sample_00.mid
```

Or use any DAW: Ableton, Logic, FL Studio, MuseScore, etc.

## Report Generation

Compile LaTeX report:
```bash
cd report
pdflatex final_report.tex
bibtex final_report
pdflatex final_report.tex
pdflatex final_report.tex
```

Output: `report/final_report.pdf`

## Creating Submission Archive

```bash
python main.py --archive
```

Creates: `music_generation_submission.zip` (ready for course submission)

## Performance Tips

1. **Use GPU**: Essential for reasonable training times
2. **Reduce dataset**: Use subset of Lakh MIDI for faster iteration
3. **Disable checkpoints**: Comment out saving for development
4. **Batch size**: Larger = faster per-epoch, but needs more memory
5. **Precision**: Use fp16 mixed precision for speed
   ```python
   from torch.cuda.amp import autocast
   with autocast():
       output = model(x)
   ```

## Reproducibility

All experiments use:
- **Random seed**: 42
- **Deterministic CUDA**: `torch.cuda.manual_seed_all(42)`
- **Fixed data splits**: Saved in `data/train_test_split/`

To reproduce exact results:
```python
import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
```

## Development Notes

### Adding New Metrics

Edit `src/evaluation/metrics.py`:
```python
class MusicMetrics:
    @staticmethod
    def new_metric(piano_roll):
        # Implement metric
        return score
```

### Custom Model Architecture

Add to `src/models/`:
```python
class CustomModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Architecture
    
    def forward(self, x):
        # Forward pass
        return output
```

### Add New Task

1. Create `src/models/task5_xxx.py`
2. Create `src/training/train_task5.py`
3. Update `main.py` pipeline

## Support & References

- **PyTorch Docs**: https://pytorch.org/docs
- **Pretty MIDI**: https://craffel.github.io/pretty-midi/
- **Transformer Paper**: https://arxiv.org/abs/1706.03762
- **VAE Paper**: https://arxiv.org/abs/1312.6114
- **RLHF Paper**: https://arxiv.org/abs/1706.03762

## Citation

If you use this code in research:

```bibtex
@misc{music_generation_2026,
    title={Unsupervised Neural Networks for Multi-Genre Music Generation},
    author={CSE425/EEE474 Neural Networks},
    year={2026}
}
```

---

**Last Updated**: May 2026
**Tested On**: Python 3.10, PyTorch 2.0, CUDA 11.8
