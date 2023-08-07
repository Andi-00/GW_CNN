#!/usr/local_rwth/bin/zsh


### TIME 
#SBATCH --time=2:00:00

### ACCOUNT
###SBATCH --account=rwth0754

### JOBNAME
#SBATCH --job-name=to0

### OUTPUT
#SBATCH --output=./to0_output.txt

### your code goes here

### Insert this AFTER the #SLURM argument section of your job script
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

### Now you can activate your configured conda environments
conda activate EMRI_env

date

python ./n1_to_0.py

date
