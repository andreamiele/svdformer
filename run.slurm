#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

module load gcc cuda cudnn mvapich2
source /ssoft/spack/external/tensorflow/tensorflow-1.3.0/tensorflow_venv_gpu/bin/activate
srun python convolutional_network.py
deactivate