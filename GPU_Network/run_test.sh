#!/usr/local_rwth/bin/zsh

### General setup for training Neural Networks with Tensorflow
### Author: Kathrin Nippel
### #SBATCH directives need to be in the first part of the jobscript

### Time
#SBATCH --time=00:20:00

### GPU
#SBATCH --gres=gpu:volta:1

# name the job
#SBATCH --job-name=train_DMNet_dbar
#SBATCH --account=rwth0754
#SBATCH --partition=c18g
#SBATCH --mem-per-cpu=100G


#SBATCH --output=./test.txt


### your code goes here, the second part of the jobscript

### begin of executable commands

date
module load GCCcore/.9.3.0
module load Python/3.9.6
module load cuDNN/8.1.1.33-CUDA-11.2.1

export TF_CPP_MIN_LOG_LEVEL="2"

python3 -c "import tensorflow as tf; print(tf.__version__)"
python3 -c "import tensorflow as tf; print(\"Num GPUs Available: \", len(tf.config.list_physical_devices('GPU')))"
pyhton3 -c "import tensorflow as tf; print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))"
date
