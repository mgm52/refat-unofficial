# ReFAT: Refusal Feature Adversarial Training (Unofficial)

This repository contains an unofficial implementation of the ReFAT (Refusal Feature Adversarial Training) paper. This uses Gemma 2 9B Instruct model and the Mechanistic-Anomaly-Detection/gemma2-jailbreaks dataset.

## Usage

You can install ReFAT as a library:

```bash
# Install directly from the repository
pip install git+https://github.com/mgm52/refat-unofficial.git

# Or, for development
git clone https://github.com/mgm52/refat-unofficial.git
cd refat-unofficial
pip install -e .
# Run the training script
python -m refat.scripts.run_refat.py
```

The package is named `refat-unofficial` but you'll still import it using the `refat` namespace:

```python
from refat.pipeline.refat import Refat, RefatConfig
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
