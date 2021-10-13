#!/bin/bash
#SBATCH --job-name=R2R-DropEnv-ObjAttn-BT_Envdrop
#SBATCH --ntasks=1                  # Run only one task
#SBATCH --output=/home/mrearle/repos/R2R-EnvDrop-ObjAttn/slurm/logs/out-%j.log    # Output name (%j is replaced by job ID)
#SBATCH --error=/home/mrearle/repos/R2R-EnvDrop-ObjAttn/slurm/logs/err-%j.log     # Output errors (optional)
#SBATCH --partition=all
#SBATCH --nodelist=scylla
#SBATCH --workdir=/home/mrearle/repos/R2R-EnvDrop-ObjAttn   # Where to run the job
#SBATCH --gres=gpu
#SBATCH --time=1-00:00
#SBATCH --mem=16gb

pwd; hostname; date

source /home/mrearle/venvs/r2r/bin/activate

echo "Starting agent training"
bash /home/mrearle/repos/R2R-EnvDrop-ObjAttn/run/bt_envdrop.bash
