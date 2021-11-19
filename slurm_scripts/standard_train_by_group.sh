#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=4000  # memory in Mb
#SBATCH --time=0-08:00:00
#SBATCH --mail-user=s1308389@ed.ac.uk
#SBATCH --mail-type=ALL

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

# Create directory on scratch for storing the model
# mkdir -p /disk/scratch/${STUDENT_ID}
export TMP=/disk/scratch/${STUDENT_ID}

# Get the parameters
while getopts g:m:s:p: option; do
    case "${option}" in
        g) group=${OPTARG};;
        m) models=${OPTARG};;
        s) script=${OPTARG};;
        p) pyth_args=${OPTARG};;
    esac
done

if [ -z ${group} ]; then
echo "Group parameter -g not given."
exit 1
fi

if [ -z ${models} ]; then
echo "Model file parameter -m not given."
exit 1
fi

if [ -z ${script} ]; then
echo "Script parameter -s not given."
exit 1
fi

# Select relevant 
MODEL="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${models}`"

config_file="${group}/${MODEL}.json"

COMMAND="sh $script -c ${config_file}"
if [ ! -z "${pyth_args}" ]; then
    COMMAND+=" -p \"${pyth_args}\""
fi
eval $COMMAND
