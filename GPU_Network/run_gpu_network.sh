#!/usr/local_rwth/bin/zsh


### TIME 
#SBATCH --time=8:00:00

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

### Insert this AFTER the #SLURM argument section of your job script
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

### Now you can activate your configured conda environments
conda activate tf
module load GCCcore/.9.3.0
module load cuDNN/8.1.1.33-CUDA-11.2.1

date

python ./network.py

date
