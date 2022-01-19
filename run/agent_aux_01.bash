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

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
