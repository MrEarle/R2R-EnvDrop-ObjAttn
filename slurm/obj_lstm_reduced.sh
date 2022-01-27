#!/bin/bash
#SBATCH --job-name=obj_lstm_red
#SBATCH --ntasks=1                  # Run only one task
#SBATCH --output=/home/mrearle/repos/R2R-EnvDrop-ObjAttn/slurm/logs/obj_lstm_red-%j.log    # Output name (%j is replaced by job ID)
#SBATCH --partition=ialab-high
#SBATCH --nodelist=grievous
#SBATCH --workdir=/home/mrearle/repos/R2R-EnvDrop-ObjAttn   # Where to run the job
#SBATCH --gres=gpu
#SBATCH --time=3-00:00

pwd; hostname; date

source /home/mrearle/venvs/r2r/bin/activate

export HDF5_USE_FILE_LOCKING="FALSE"
echo "Starting agent training"
name="obj_vis_lstm"
flag="--attn soft --train listener 
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --include_objs
      --include_objs_lstm
      --reduced_envs
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 120000 --maxAction 35"
mkdir -p snap/$name
python r2r_src/train.py $flag --name $name
