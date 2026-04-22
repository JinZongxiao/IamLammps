#!/bin/bash
#SBATCH --job-name=w_tensile
#SBATCH --output=/ssd1/jinzongxiao/iam_runs/logs/w_tensile_%j.out
#SBATCH --error=/ssd1/jinzongxiao/iam_runs/logs/w_tensile_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# ============================================
# IamLAMMPS Tensile Production Job Script
# ============================================

set -e

# Ensure directories exist
mkdir -p /ssd1/jinzongxiao/iam_runs/logs
mkdir -p /ssd1/jinzongxiao/iam_runs/runs

# Activate conda environment
source /home/jinzongxiao/miniforge/bin/activate iam-lammps

# Set environment variables
export LAMMPS_COMMAND=lmp
export LAMMPS_POTENTIALS_DIR=$CONDA_PREFIX/share/lammps/potentials
export RUNS_DIR=/ssd1/jinzongxiao/iam_runs/runs
export MEMORY_FILE=/ssd1/jinzongxiao/iam_runs/simulation_memory.jsonl

# Load API config from .env if present
if [ -f /ssd1/jinzongxiao/IamLAMMPS/.env ]; then
    export $(grep -v '^#' /ssd1/jinzongxiao/IamLAMMPS/.env | xargs)
fi

# Run tensile workflow
cd /ssd1/jinzongxiao/IamLAMMPS
python -m src.lammps_agent.main \
    --workflow tensile \
    --runs-dir /ssd1/jinzongxiao/iam_runs/runs \
    --supercell 10,10,20 \
    --temperature 300.0 \
    --equil-steps 20000 \
    --timestep 0.001 \
    --thermo-every 100 \
    --dump-every 1000 \
    --run
