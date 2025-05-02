#!/bin/bash
#SBATCH --partition=general        # Request partition (adjust as needed)
#SBATCH --qos=short                # Request Quality of Service (adjust as needed)
#SBATCH --time=1:00:00             # Request run time (adjust as needed)
#SBATCH --ntasks=1                 # Request number of parallel tasks per job
#SBATCH --cpus-per-task=2          # Request number of CPUs (adjust as needed)
#SBATCH --mem=4096                 # Request memory (adjust as needed)
#SBATCH --mail-type=END            # Send email when job finishes
#SBATCH --output=slurm_%j.out      # Output log file
#SBATCH --error=slurm_%j.err       # Error log file

# Load Conda (if needed)
module load conda

# Create the conda environment from the environment.yml file
conda env create -f environment.yaml

# Activate the conda environment
conda activate jax_gpu

# Run your Python script
srun python main.py

# Optionally deactivate the environment after the script finishes
conda deactivate
