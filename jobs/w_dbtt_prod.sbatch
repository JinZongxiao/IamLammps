#!/bin/bash
#SBATCH --job-name=w_dbtt
#SBATCH --output=/ssd1/jinzongxiao/iam_runs/logs/w_dbtt_%j.out
#SBATCH --error=/ssd1/jinzongxiao/iam_runs/logs/w_dbtt_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# ============================================
# IamLAMMPS DBTT Sweep Production Job Script
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

# Run DBTT sweep workflow
cd /ssd1/jinzongxiao/IamLAMMPS
python -m src.lammps_agent.main \
    --workflow dbtt \
    --runs-dir /ssd1/jinzongxiao/iam_runs/runs \
    --supercell 10,10,20 \
    --equil-steps 10000 \
    --timestep 0.001 \
    --thermo-every 100 \
    --dump-every 2000 \
    --run
