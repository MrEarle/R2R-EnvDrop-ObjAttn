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

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
