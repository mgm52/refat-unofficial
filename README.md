# ReFAT: Refusal Feature Adversarial Training (Unofficial)

This repository contains an unofficial implementation of the ReFAT (Refusal Feature Adversarial Training) paper! This implementation uses the Gemma 2 9B Instruct model and the Mechanistic-Anomaly-Detection/gemma2-jailbreaks dataset, which contains both harmful jailbreak attempts and harmless instructions, adapted from Circuit Breakers.

## Getting Started

```bash
# Install dependencies (same as the obfuscated-activations repo)
./install.sh

# Run the training script
python refat/scripts/train.py
```

## Project Structure

```
refat/
├── model_utils/       # Model-specific utilities
├── pipeline/          # Core training pipeline
│   └── utils/         # General utilities for the pipeline
├── data_utils/        # Dataset loading and processing
└── scripts/           # Training and evaluation scripts
```