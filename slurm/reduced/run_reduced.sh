#!/bin/bash

#sbatch slurm/reduced/base_reduced.sh
# sbatch slurm/reduced/objs_reduced.sh
sbatch slurm/reduced/objs_reduced_aux01.sh
sbatch slurm/reduced/objs_reduced_aux10.sh
# sbatch 'slurm/reduced/more_obj(32)-aux01_reduced.sh'
# sbatch 'slurm/reduced/more_obj(64)-aux01_reduced.sh'
