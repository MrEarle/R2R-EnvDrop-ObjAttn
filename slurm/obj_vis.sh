#!/bin/bash
#SBATCH --job-name=obj_vis
#SBATCH --ntasks=1                  # Run only one task
#SBATCH --output=/home/mrearle/repos/R2R-EnvDrop-ObjAttn/slurm/logs/obj_vis-%j.log    # Output name (%j is replaced by job ID)
#SBATCH --partition=ialab-high
#SBATCH --nodelist=scylla
#SBATCH --workdir=/home/mrearle/repos/R2R-EnvDrop-ObjAttn   # Where to run the job
#SBATCH --gres=gpu
#SBATCH --time=3-00:00

pwd; hostname; date

source /home/mrearle/venvs/r2r/bin/activate

export HDF5_USE_FILE_LOCKING="FALSE"
echo "Starting agent training"
name="agent_obj_visual_matt"
flag="--attn soft --train listener 
      --featdropout 0.3
      --load snap/agent_obj_visual_matt/state_dict/best_val_unseen
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --include_objs
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 120000 --maxAction 35"
mkdir -p snap/$name
python r2r_src/train.py $flag --name $name
