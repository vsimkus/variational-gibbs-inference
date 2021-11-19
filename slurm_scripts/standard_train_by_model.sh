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
mkdir -p /disk/scratch/${STUDENT_ID}
export TMP=/disk/scratch/${STUDENT_ID}

# Get the parameters
while getopts b:g:m:s:p: option; do
    case "${option}" in
        b) group_base=${OPTARG};;
        g) groups=${OPTARG};;
        m) model=${OPTARG};;
        s) script=${OPTARG};;
        p) pyth_args=${OPTARG};;
    esac
done

if [ -z ${group_base} ]; then
echo "Group base parameter -b not given."
exit 1
fi

if [ -z ${model} ]; then
echo "Model parameter -m not given."
exit 1
fi

if [ -z ${groups} ]; then
echo "Groups parameter -g not given."
exit 1
fi

if [ -z ${script} ]; then
echo "Script parameter -s not given."
exit 1
fi

# Parse groups into an array
IFS=',' read -r -a groups <<< "${groups}"

# Select relevant 
GROUP=${groups[$SLURM_ARRAY_TASK_ID]}
group="${group_base}/${GROUP}"

config_file="${group}/${model}.json"

COMMAND="sh $script -c ${config_file}"
if [ ! -z "${pyth_args}" ]; then
    COMMAND+=" -p \"${pyth_args}\""
fi
eval $COMMAND
