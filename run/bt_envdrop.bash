name=agent_bt_obj_pruned_mask
# aug: the augmented paths, only the paths are used (not the insts)
# speaker: load the speaker from
# load: load the agent from
flag="--attn soft --train auglistener --selfTrain
      --aug tasks/R2R/data/aug_paths.json
      --speaker snap/speaker_obj_pruned/state_dict/best_val_unseen_bleu 
      --load snap/agent_obj_pruned_mask/state_dict/best_val_unseen
      --angleFeatSize 128
      --accumulateGrad
      --featdropout 0.4
      --subout max --optim rms --lr 1e-4 --iters 300000 --maxAction 35"
mkdir -p snap/$name
python r2r_src/train.py $flag --name $name 

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
