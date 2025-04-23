#!/bin/bash
#SBATCH --account=mi2lab-normal
#SBATCH --job-name=borf
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dawid.pludowski@gmail.com
#SBATCH --output=/mnt/evafs/groups/mi2lab/dpludowski/code/hellsemble/logs/eden-%A-%a.log
#SBATCH --error=/mnt/evafs/groups/mi2lab/dpludowski/code/hellsemble/logs/eden-%A-%a-error.log


. /mnt/evafs/groups/mi2lab/dpludowski/miniconda3/etc/profile.d/conda.sh
conda activate hellsemble

export PYTHONPATH=`pwd`

FILE_LIST=($(ls "resources/data/openml-restructured"))
FILE_NAME=${FILE_LIST[$SLURM_ARRAY_TASK_ID]}


echo "Used data: $FILE_NAME"
mkdir -p "resources/results/$FILE_NAME"

# Run your Python script with this file
python bin/run_experiment.py \
    --train-dir="resources/data/openml-restructured/$FILE_NAME/train" \
    --test-dir="resources/data/openml-restructured/$FILE_NAME/test" \
    --output-dir="resources/results/$FILE_NAME"
