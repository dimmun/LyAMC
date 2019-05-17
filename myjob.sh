#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --time=1:00:00
#SBATCH --export=all
#SBATCH --mem=50gb

pwd
python test.py