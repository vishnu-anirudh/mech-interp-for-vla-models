# Disentangling Physics from Pixels: Robust Feature Steering in Vision-Language-Action Models

Research project on mechanistic interpretability for vision-language-action (VLA) models, focusing on discovering and steering physical dynamics features (e.g., fragility, mass) in robotics applications.

## Overview

This project implements a complete pipeline for:
- **Feature Discovery**: Using Matryoshka Sparse Autoencoders (MSAE) to discover interpretable features in VLM activations
- **Decorrelated Analysis**: Identifying genuine physical properties (fragility) vs. visual confounders (color)
- **Feature Steering**: Modifying model behavior by injecting discovered features
- **World Model Probes**: Evaluating anticipatory physics understanding
- **Adversarial Defense**: SAE-based anomaly detection (SARM) for secure steering

## Key Results

- **Patch-based training**: Scaled dataset by 100× (4k images → 400k patch tokens)
- **Model capacity threshold**: Established that 2.25B models lack robust physics understanding (R² = -2.08)
- **MSAE steering**: Demonstrated technical feasibility with measurable output shifts
- **Scalable anomaly detection**: Implemented batch processing for 400k+ patch tokens

## Installation

### Requirements

- Python 3.14+
- PyTorch 2.9.1+ (with MPS support for Mac)
- PyBullet 3.2.7+
- Transformers 4.57.3+

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd mechanistic_interpretability
   ```

2. **Install dependencies using `uv`:**
   ```bash
   uv pip install -e .
   ```

   For OpenVLA support (optional):
   ```bash
   uv pip install -e ".[openvla]"
   ```

   For development (includes ruff linter):
   ```bash
   uv pip install -e ".[dev]"
   ```

   This will install all dependencies specified in `pyproject.toml`. If you need to use `pip` instead, you can generate a `requirements.txt` file:
   ```bash
   uv pip compile pyproject.toml -o requirements.txt
   pip install -r requirements.txt
   ```

3. **Download PhysObjects dataset (optional):**
   - Download from: https://iliad.stanford.edu/pg-vlm/
   - Extract to `data/physobjects/physobjects/`

## Usage

### Basic Usage

Run the full pipeline with default settings (SmolVLM, 4,000 samples):

```bash
python main.py
```

### Command-Line Options

```bash
python main.py [OPTIONS]

Options:
  --model {smolvlm,openvla}    VLM model to use (default: smolvlm)
  --num_samples N               Number of samples to collect (default: 4000)
  --num_epochs E                MSAE training epochs (default: 200)
  --use_physobjects             Use PhysObjects dataset (requires EgoObjects images)
  --physobjects_path PATH       Path to PhysObjects dataset
  --egoobjects_path PATH        Path to EgoObjects images (required if using PhysObjects)
  --checkpoint_dir PATH         Directory for checkpoints (default: checkpoints/)
  --no-resume                   Force regeneration, ignore checkpoints
  --force_cpu_training          Force CPU for MSAE training (avoids MPS 4GB limit)
  --skip_validation             Skip model capacity validation
```

### Examples

**Run with SmolVLM (faster, 2.25B):**
```bash
python main.py --model smolvlm --num_samples 4000
```

**Run with OpenVLA (slower, better physics, 7B):**
```bash
python main.py --model openvla --num_samples 4000 --force_cpu_training
```

**Use PhysObjects dataset:**
```bash
python main.py --use_physobjects --physobjects_path data/physobjects/physobjects --egoobjects_path /path/to/egoobjects
```

**Resume from checkpoint:**
```bash
python main.py --checkpoint_dir checkpoints/
```

## Project Structure

```
mechanistic_interpretability/
├── main.py                    # Entry point for main pipeline
├── generate_graphs.py         # Entry point for graph generation
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── main.py               # Main pipeline script
│   ├── sae.py                # Standard Sparse Autoencoder
│   ├── msae.py               # Matryoshka Sparse Autoencoder (nested structure)
│   ├── vlm_wrapper.py        # Vision-Language Model wrapper (SmolVLM/OpenVLA)
│   ├── environment.py        # PyBullet robotics simulation
│   ├── hooks.py              # Activation hooks (data collection, steering)
│   ├── hooks_patch_based.py  # Patch-based activation extraction
│   ├── dataset_loader.py    # Dataset loading (simulation/PhysObjects)
│   ├── world_model_probe.py  # World model probe for anticipatory physics
│   └── generate_graphs_from_results.py  # Graph generation utility
│
├── reports/                   # Research reports
│   ├── APPLICATION_REPORT_PUBLISHABLE.md   # Main research report
│   ├── APPLICATION_REPORT_NOTION.md       # Notion-formatted report
│   └── APPLICATION_REPORT_GOOGLEDOCS.html  # Google Docs formatted report
│
├── graphs/                     # Visualization outputs
│   ├── graph1_decorrelated_analysis.png    # Feature activation analysis
│   ├── graph2_msae_training.png            # MSAE training curves
│   ├── graph3_msae_steering.png           # Steering comparison
│   ├── graph4_world_model_probe.png       # World model probe results
│   └── graph5_sarm_defense.png            # SARM defense results
│
├── docs/                       # Documentation
│   ├── NOTION_IMPORT_GUIDE.md
│   └── CLEANUP_SUMMARY.md
│
├── data/                       # Data directory
│   └── physobjects/           # PhysObjects dataset (optional)
│
├── checkpoints/                # Training checkpoints (auto-saved)
├── graph_data/                 # Saved data for graph generation
│
├── pyproject.toml             # Project configuration and dependencies (uv)
├── uv.lock                    # Locked dependency versions
├── .gitignore                 # Git ignore rules
└── README.md                   # This file
```

## Core Components

### 1. Matryoshka Sparse Autoencoder (MSAE)

Nested SAE architecture that separates coarse semantics from fine syntax:
- **Coarse features** (first 512 dims): High-level semantics (fragility, material)
- **Fine features** (remaining dims): Syntax and grammar

This enables steering semantics without disrupting grammar.

**File:** `msae.py`

### 2. Vision-Language Model Wrapper

Unified interface for different VLMs:
- **SmolVLM** (2.25B): Fast, good for proof-of-concept
- **OpenVLA** (7B): Better physics understanding, slower

Automatically detects architecture and handles device placement (MPS/CUDA/CPU).

**File:** `vlm_wrapper.py`

### 3. Decorrelated Data Collection

Addresses the "Clever Hans" problem by decorrelating visual appearance from physical properties:
- Red/Fragile, Red/Rigid, Blue/Fragile, Blue/Rigid (balanced)
- Identifies genuine fragility features vs. color confounders

**Files:** `dataset_loader.py`, `environment.py`

### 4. Patch-Based Training

Extracts all visual patch tokens (not just last token) to scale dataset by 100×:
- 4,000 images → 400,000 patch tokens
- Enables robust SAE training on large-scale data

**File:** `hooks_patch_based.py`

### 5. World Model Probe

Linear probe to predict future states ($S_{t+1}$) from current activations ($h_t$):
- Tests anticipatory physics understanding
- Establishes model capacity requirements

**File:** `world_model_probe.py`

## Generating Graphs

After running the pipeline, generate visualization graphs:

```bash
python generate_graphs.py
```

This will create graphs in the `graphs/` directory:
- `graphs/graph1_decorrelated_analysis.png`: Feature activation by object type
- `graphs/graph2_msae_training.png`: Training loss curves
- `graphs/graph3_msae_steering.png`: Steering output comparison
- `graphs/graph4_world_model_probe.png`: World model probe results
- `graphs/graph5_sarm_defense.png`: SARM defense threshold analysis

## Research Reports

Three formatted versions of the research report are included in the `reports/` directory:

1. **reports/APPLICATION_REPORT_PUBLISHABLE.md**: Main markdown report
2. **reports/APPLICATION_REPORT_NOTION.md**: Notion-optimized format
3. **reports/APPLICATION_REPORT_GOOGLEDOCS.html**: Google Docs format

See `docs/NOTION_IMPORT_GUIDE.md` for import instructions.

## Checkpointing

The pipeline automatically saves checkpoints at key stages:
- Dataset collection
- MSAE training (every 50 epochs)
- Feature discovery
- Steering results
- SARM defense
- World model probe

To resume from a checkpoint:
```bash
python main.py  # Automatically resumes if checkpoints exist
```

To force regeneration:
```bash
python main.py --no-resume
```

## Development

### Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting.

**Check for linting issues:**
```bash
ruff check .
```

**Auto-fix linting issues:**
```bash
ruff check --fix .
```

**Format code:**
```bash
ruff format .
```

**Check and format in one command:**
```bash
ruff check --fix . && ruff format .
```

## Hardware Requirements

### Minimum
- CPU: 4+ cores
- RAM: 16GB
- Storage: 10GB (for models and data)

### Recommended
- **Mac with Apple Silicon**: MPS acceleration (10-50× speedup)
- **GPU**: CUDA-compatible GPU (8GB+ VRAM)
- RAM: 32GB+ for large datasets
- Storage: 50GB+ for full PhysObjects dataset

### Memory Management

- **MPS 4GB limit**: On Mac, MPS has a 4GB limit. The code automatically switches to CPU for large training batches.
- **Force CPU training**: Use `--force_cpu_training` to avoid MPS limits
- **Batch processing**: Large datasets are automatically processed in batches

## Troubleshooting

### OpenVLA Loading Issues

If OpenVLA fails to load due to TIMM version, install the optional dependencies:
```bash
uv pip install -e ".[openvla]"
```

Or with pip:
```bash
pip install "timm>=0.9.10,<1.0.0"
```

### Out of Memory Errors

- Use `--force_cpu_training` for MSAE training
- Reduce `--num_samples` if dataset collection fails
- Use smaller batch sizes (modify code if needed)

### Slow Performance

- **SmolVLM**: Faster but limited physics understanding
- **OpenVLA**: Slower but better results
- **MPS**: Ensure MPS is available (Mac only)
- **CPU**: Consider using a GPU or cloud instance

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mechanistic_interpretability_vla,
  title={Disentangling Physics from Pixels: Robust Feature Steering in Vision-Language-Action Models},
  author={[Your Name]},
  year={2025},
  journal={[Journal/Conference]}
}
```

## License

[Specify your license here]

## Contact

[Your contact information]

## Acknowledgments

- **PhysObjects Dataset**: For physical dynamics annotations
- **HuggingFace**: For SmolVLM and OpenVLA models
- **PyBullet**: For robotics simulation
- **Transformers**: For model loading and processing

## References

1. Häon et al. (2025). "Mechanistic Interpretability for Steering Vision-Language-Action Models."
2. Zaigrajew et al. (2025). "Matryoshka Sparse Autoencoders."
3. PhysObjects (2024). "Physically Grounded Vision-Language Models for Robotic Manipulation."

