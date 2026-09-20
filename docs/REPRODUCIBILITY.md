# Experiment settings

This guide summarizes the settings reported in the SIGNAL paper and the repository’s configuration conventions.

## Environment

The paper reports an NVIDIA RTX A40 GPU with 48 GB memory and a PyTorch implementation.

The supplied [environment.yml](../environment.yml) uses Python 3.9 and preserves the original repository's main dependency pins: NumPy 1.21.5, pandas 1.3.5, SciPy 1.7.3, and PyTorch 1.11.0. Python is specified in the Conda environment rather than as a pip package. Dependencies directly imported by the existing loader and trainer (torchvision, Pillow, LMDB, and matplotlib) have been added to [requirements.txt](../requirements.txt).

```bash
conda env create -f environment.yml
conda activate signal
```

The implementation uses CUDA for graph construction and 64-dimensional embeddings.

## Settings reported by the paper

Section 4.1.4 specifies the following:

| Setting | Paper value |
| :--- | :--- |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 4096 |
| Regularization coefficient | 0.01 |
| User–item convolution layers | 2 |
| Item–item convolution layers | 1 |
| Co-occurrence pruning threshold | 2 |
| Early-stopping patience | 25 epochs |
| Interaction split | 80% / 10% / 10% |
| Evaluation | Recall and NDCG at K = 10 and 20 |

The framework-level defaults are in `src/configs/overall.yaml`. Dataset and model settings override these defaults according to the loading order below.

Table 7 reports these selected hyperparameters:

| Dataset | λᵇ_CL | λᵈ_CL | k | ρ | μ | λ_DM | α₁ | α₂ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baby | 0.01 | 0.001 | 10 | 5 | 0.4 | 10 | 0.6 | 0.1 |
| Sports | 0.01 | 0.001 | 5 | 6 | 0.5 | 1 | 0.5 | 0.1 |
| Clothing | 0.01 | 0.001 | 5 | 5 | 0.6 | 10 | 0.5 | 0.1 |
| TikTok | 0.01 | 0.001 | 5 | 5 | 0.8 | 10 | 0.6 | 0.1 |

The values above are transcribed from Table 7 of the paper.

## Training entry point

The training entry point accepts the model and dataset names. From the repository root:

```bash
cd src
python main.py --model SIGNAL --dataset baby
```

The loader merges the following in order:

1. `src/configs/overall.yaml`.
2. `src/configs/dataset/baby.yaml`.
3. A model-specific YAML at `src/configs/model/<model>.yaml`, when present.
4. The explicit configuration dictionary in `main.py`.

`main.py` accepts `--model` / `-m` and `--dataset` / `-d`. Other command-line options are ignored by the existing parser; change YAML or the explicit configuration dictionary for other settings. Start from `src/` because configuration and data paths are relative to the working directory.
