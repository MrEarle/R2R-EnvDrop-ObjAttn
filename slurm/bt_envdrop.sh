#!/bin/bash
#SBATCH --job-name=R2R-DropEnv-ObjAttn-BT_Envdrop
#SBATCH --ntasks=1                  # Run only one task
#SBATCH --output=/home/mrearle/repos/R2R-EnvDrop-ObjAttn/slurm/logs/envdrop_out.log    # Output name (%j is replaced by job ID)
#SBATCH --error=/home/mrearle/repos/R2R-EnvDrop-ObjAttn/slurm/logs/envdrop_err.log     # Output errors (optional)
#SBATCH --partition=ialab-high
#SBATCH --nodelist=scylla
#SBATCH --workdir=/home/mrearle/repos/R2R-EnvDrop-ObjAttn   # Where to run the job
#SBATCH --gres=gpu:TitanRTX
#SBATCH --time=2-00:00
#SBATCH --mem=20gb

pwd; hostname; date

source /home/mrearle/venvs/r2r/bin/activate

echo "Starting envdrop training"
bash /home/mrearle/repos/R2R-EnvDrop-ObjAttn/run/bt_envdrop.bash
