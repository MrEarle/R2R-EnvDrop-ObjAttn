name=speaker_rxr
flag="--attn soft --angleFeatSize 128
      --train speaker
      --speaker snap/speaker_rxr/state_dict/best_val_unseen_loss
      --subout max --dropout 0.6 --optim adam --lr 1e-4 --iters 100000 --maxAction 35"
mkdir -p snap/$name
python r2r_src/train.py $flag --name $name 

# Try this for file logging
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
