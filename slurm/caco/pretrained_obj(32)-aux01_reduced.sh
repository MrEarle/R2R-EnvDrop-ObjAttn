#!/bin/bash
#SBATCH --job-name=pretrain-obj32-aux01
#SBATCH --ntasks=1                  # Run only one task
#SBATCH --output=/home/mrearle/repos/R2R-EnvDrop-ObjAttn/slurm/logs/pretrain-obj32-aux01-%j.log    # Output name (%j is replaced by job ID)
#SBATCH --partition=ialab-high
#SBATCH --nodelist=scylla
#SBATCH --workdir=/home/mrearle/repos/R2R-EnvDrop-ObjAttn   # Where to run the job
#SBATCH --mem=32gb
#SBATCH --gres=gpu:TitanRTX
#SBATCH --time=3-00:00
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=biearle@uc.cl     # Where to send mail	

pwd; hostname; date

source /home/mrearle/venvs/r2r/bin/activate

export HDF5_USE_FILE_LOCKING="FALSE"
echo "Starting agent training"
# bash /home/mrearle/repos/R2R-EnvDrop-ObjAttn/run/agent_aux_01.bash 0
name="craft_pretrain/obj"
flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --include_objs
      --max_obj_number 32
      --obj_aux_task
      --obj_aux_task_weight 0.1
      --reduced_envs
      --dataset r2r
      --load snap/craft_pretrain/obj/obj(32)_aux(0.1)_reduced/state_dict/latest_iter
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 200000 --maxAction 35"
mkdir -p snap/$name
python r2r_src/train.py $flag --name $name
