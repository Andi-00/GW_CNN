#!/usr/local_rwth/bin/zsh


### TIME 
#SBATCH --time=20:00:00

### ACCOUNT
###SBATCH --account=rwth0754

### JOBNAME
#SBATCH --job-name=network_run

### OUTPUT
#SBATCH --output=./Network/network_output/run_5/run_5.txt

### your code goes here

### Insert this AFTER the #SLURM argument section of your job script
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

### Now you can activate your configured conda environments
conda activate tf

date

python ./network.py

date