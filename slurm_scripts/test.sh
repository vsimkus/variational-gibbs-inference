#!/bin/sh

# Get the parameters
while getopts c:p: option; do
    case "${option}" in
        c) config=${OPTARG};;
        p) pyth_args=${OPTARG};;
    esac
done

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate cdi
cd ..

json_config="experiment_configs/${config}"
echo "Starting test.py on ${config}"
echo "Additional python args: ${pyth_args}"

# Read experiment directory name from the config json
experiment_name=$(python -c "from test import build_argparser; \
                             print(build_argparser().parse_args().experiment_name)" \
                  --test_config=$json_config \
                  ${pyth_args})
exp_group=$(python -c "from test import build_argparser; \
                       print(build_argparser().parse_args().exp_group)" \
            --test_config=$json_config \
            ${pyth_args})
full_exp_name=$(python -c "from test import build_argparser; \
                           from cdi.util.utils import construct_experiment_name; \
                           print(construct_experiment_name(build_argparser().parse_args()))" \
                --test_config=$json_config \
                ${pyth_args})


mkdir -p trained_models/${exp_group}/${full_exp_name}

if [ -z ${full_exp_name} ]; then
echo "No experiment name!"
exit 1
fi

# Remove the experiment directory if already exists
if [ -d "${TMP}/trained_models/${exp_group}/${full_exp_name}" ];
then
    rm -r "${TMP}/trained_models/${exp_group}/${full_exp_name}"
fi

# export PYTHONUNBUFFERED=TRUE # This allows to dump the log messages into stdout immediately
COMMAND="python test.py --gpus='0' --test_config=$json_config --output_root_dir=${TMP} ${pyth_args}"
eval $COMMAND

dest="trained_models/${exp_group}/${experiment_name}"
echo "Copying evaluation results to head node at $dest."
cp -r "${TMP}/trained_models/${exp_group}/${full_exp_name}" "trained_models/${exp_group}/${experiment_name}"

# Remove the experiment directory
rm -r "${TMP}/trained_models/${exp_group}/${full_exp_name}"
