# ECG Analysis with Large Language Models: ICL and CBM Study

## Overview

This project explores the application of In-Context Learning (ICL) to electrocardiogram (ECG) analysis using multimodal large language models. The main challenge addressed is adapting open-source models to medical imaging tasks without prior exposure to ECG data distribution, while maintaining data privacy by avoiding external APIs.

## Motivation

While powerful proprietary models like GPT-4o demonstrate strong performance on ECG interpretation tasks (likely due to training exposure), their use requires sending sensitive medical data to external servers. This project investigates how far we can push open-source models (e.g., MedGemma) using ICL combined with Concept Bottleneck Models (CBM) and targeted fine-tuning.

## Research Hypothesis

- ICL performance degrades when data distribution differs significantly from the model's training data
- Simple fine-tuning on basic ECG concepts (P, Q, R, S, T waves, grid scale) can overcome this limitation
- CBM can improve interpretability and performance by introducing explicit concept reasoning

## Experimental Pipeline

1. Baseline: MedGemma with ICL
2. MedGemma with ICL + CBM
3. Fine-tune on basic ECG concepts (without diagnostic labels)
4. Fine-tuned model with ICL
5. Fine-tuned model with ICL + CBM

## Installation

```bash
# Create environment
conda env create -f environment.yml
conda activate ecg-icl

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Preprocess data
python scripts/preprocess_ptbxl.py --data_dir data/raw/PTBXL

# Run toy experiment
python scripts/run_experiment.py --config configs/toy_experiment.yaml

# Run ICL evaluation
python scripts/run_experiment.py --config configs/medgemma_icls.yaml

# Train CBM
python scripts/train_cbm.py --config configs/cbm_config.yaml
```

## Project Structure

- `configs/` - Experiment configuration files
- `data/` - Dataset storage (not tracked by git)
- `notebooks/` - Exploratory analysis and visualization
- `scripts/` - Command-line executable scripts
- `src/` - Core source code modules
- `experiments/` - Training logs and checkpoints
- `results/` - Final figures and tables
- `docs/` - Technical documentation

## Privacy and Ethics

This project prioritizes data privacy by focusing on open-source models that can be deployed locally in hospital environments, avoiding the need to transmit sensitive medical data to external APIs.

## License

TBD

## Citation

TBD

