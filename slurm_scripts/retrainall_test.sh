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
echo "Starting retrain_all_ckpts_on_test_and_run_test.py on ${config}"
echo "Additional python args: ${pyth_args}"

# Read experiment directory name from the config json
experiment_name=$(python -c "from retrain_all_ckpts_on_test_and_run_test import build_argparser; \
                             print(build_argparser()._parse_known_args()[0].experiment_name)" \
                  --config=$json_config \
                  ${pyth_args})
exp_group=$(python -c "from retrain_all_ckpts_on_test_and_run_test import build_argparser; \
                       print(build_argparser()._parse_known_args()[0].exp_group)" \
            --config=$json_config \
            ${pyth_args})
# full_exp_name=$(python -c "from retrain_all_ckpts_on_test_and_run_test import build_argparser; \
#                            from cdi.util.utils import construct_experiment_name; \
#                            print(construct_experiment_name(build_argparser().parse_args()))" \
#                 --config=$json_config \
#                 ${pyth_args})

mkdir -p trained_models/${exp_group}/${experiment_name}

if [ -z ${experiment_name} ]; then
echo "No experiment name!"
exit 1
fi

# Remove the experiment directories if already exist
rm -r "${TMP}/trained_models/${exp_group}/${experiment_name}"* || true

# export PYTHONUNBUFFERED=TRUE # This allows to dump the log messages into stdout immediately
COMMAND="python retrain_all_ckpts_on_test_and_run_test.py --gpus='0' --config=$json_config --output_root_dir=${TMP} ${pyth_args}"
eval $COMMAND

dest="trained_models/${exp_group}/"
echo "Copying model parameters to head node at $dest. Name: ${experiment_name}"

cp -r "${TMP}/trained_models/${exp_group}/${experiment_name}"* $dest

# Remove the experiment directory
rm -r "${TMP}/trained_models/${exp_group}/${experiment_name}"*
