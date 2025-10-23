#!/bin/bash
#SBATCH --job-name=setup_ds
#SBATCH --output=setup_ds_%j.out
#SBATCH --error=setup_ds_%j.err
#SBATCH --time=5:00:00
#SBATCH --gpus=1 # This allocates 72 CPU cores
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luciarosequirke@gmail.com


cd /home/lucia/bergson
source .venv/bin/activate

# Set HF cache

# Run the Python command
HUGGINGFACE_HUB_CACHE="/projects/a5k/public/lucia" \
TRANSFORMERS_CACHE="/projects/a5k/public/lucia" \
uv run python -m examples.setup_ds
