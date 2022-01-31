#!/bin/bash

sbatch slurm/reduced/base_reduced.sh
sbatch slurm/reduced/objs_reduced.sh
sbatch slurm/reduced/objs_reduced_aux01.sh
