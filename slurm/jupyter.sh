#!/bin/bash
#SBATCH --job-name=envdrop-jupyter
#SBATCH --ntasks=1                  # Run only one task
#SBATCH --output=/home/mrearle/slurm/jupyter.log    # Output name (%j is replaced by job ID)
#SBATCH --error=/home/mrearle/slurm/jupyter.log     # Output errors (optional)
#SBATCH --partition=all
#SBATCH --nodelist=scylla
#SBATCH --workdir=/home/mrearle/repos/R2R-EnvDrop-ObjAttn   # Where to run the job
#SBATCH --gres=gpu

pwd; hostname; date
source /home/mrearle/venvs/r2r/bin/activate

echo "Starting notebook"
jupyter notebook --no-browser --port=9999 --ip="0.0.0.0" && jupyter notebook --no-browser --port=9999