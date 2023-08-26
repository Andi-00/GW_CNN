#!/usr/local_rwth/bin/zsh


### TIME 
#SBATCH --time=05:00:00

### GPU
#SBATCH --gres=gpu:volta:1

# name the job
#SBATCH --job-name=run_1.12
#SBATCH --account=rwth0754
#SBATCH --partition=c18g
#SBATCH --mem-per-cpu=100G


### OUTPUT
#SBATCH --output=./network_output/outputs/run_1.12.txt

### your code goes here



date
module load GCCcore/.9.3.0
module load Python/3.9.6
module load cuDNN/8.1.1.33-CUDA-11.2.1

export TF_CPP_MIN_LOG_LEVEL="2"

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_ROOT"


date

python3 ./network.py

date
