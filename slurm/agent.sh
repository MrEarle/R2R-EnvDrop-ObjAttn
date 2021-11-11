#!/bin/bash
#SBATCH --job-name=R2R-DropEnv-ObjAttn-Agent
#SBATCH --ntasks=1                  # Run only one task
#SBATCH --output=/home/mrearle/repos/R2R-EnvDrop-ObjAttn/slurm/logs/agent_out-%j.log    # Output name (%j is replaced by job ID)
#SBATCH --error=/home/mrearle/repos/R2R-EnvDrop-ObjAttn/slurm/logs/agent_err-%j.log     # Output errors (optional)
#SBATCH --partition=ialab-high
#SBATCH --nodelist=scylla
#SBATCH --workdir=/home/mrearle/repos/R2R-EnvDrop-ObjAttn   # Where to run the job
#SBATCH --gres=gpu
#SBATCH --time=3-00:00
#SBATCH --mem=16gb

pwd; hostname; date

source /home/mrearle/venvs/r2r/bin/activate

echo "Starting agent training"
bash /home/mrearle/repos/R2R-EnvDrop-ObjAttn/run/agent.bash 0
