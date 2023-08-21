#!/usr/local_rwth/bin/zsh


### TIME 
#SBATCH --time=02:00:00

### GPU
#SBATCH --gres=gpu:volta:1

# name the job
#SBATCH --job-name=gpu_run
#SBATCH --account=rwth0754
#SBATCH --partition=c18g
#SBATCH --mem-per-cpu=100G


### OUTPUT
#SBATCH --output=./gpu_output.txt

### your code goes here



date
module load GCCcore/.9.3.0
module load Python/3.9.6
module load cuDNN/8.1.1.33-CUDA-11.2.1

export TF_CPP_MIN_LOG_LEVEL="2"


date

python3 ./network.py

date
