#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --node=1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1

module load gcc python
python3 temp.py