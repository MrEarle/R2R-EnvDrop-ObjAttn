#!bin/bash

# sbatch slurm/caco/craft_base_reduced.sh
sbatch slurm/caco/obj_lstm_reduced.sh
sbatch slurm/caco/more_obj\(32\)-aux01_reduced.sh
# sbatch slurm/caco/more_obj\(64\)-reduced.sh
sbatch slurm/caco/more_obj\(64\)-aux01_reduced.sh
# sbatch slurm/caco/more_obj\(32\)-reduced.sh