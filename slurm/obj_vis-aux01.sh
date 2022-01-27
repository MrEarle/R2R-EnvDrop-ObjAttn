#!/bin/bash
#SBATCH --job-name=obj_vis-aux01
#SBATCH --ntasks=1                  # Run only one task
#SBATCH --output=/home/mrearle/repos/R2R-EnvDrop-ObjAttn/slurm/logs/obj_vis-aux01-%j.log    # Output name (%j is replaced by job ID)
#SBATCH --partition=ialab-high
#SBATCH --nodelist=scylla
#SBATCH --workdir=/home/mrearle/repos/R2R-EnvDrop-ObjAttn   # Where to run the job
#SBATCH --gres=gpu
#SBATCH --time=3-00:00

pwd; hostname; date

source /home/mrearle/venvs/r2r/bin/activate

export HDF5_USE_FILE_LOCKING="FALSE"
echo "Starting agent training"
# bash /home/mrearle/repos/R2R-EnvDrop-ObjAttn/run/agent_aux_01.bash 0
name="agent_obj_visual_matt"
flag="--attn soft --train listener 
      --load snap/agent_obj_visual_matt-aux(0.1)/state_dict/best_val_unseen
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --obj_aux_task
      --include_objs
      --obj_aux_task_weight 0.1
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 120000 --maxAction 35"
mkdir -p snap/$name
python r2r_src/train.py $flag --name $name