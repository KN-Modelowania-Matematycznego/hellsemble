DEFAULT_INPUT_DIR="resources/data/openml-restructured"
INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}"

NUM_FILES=$(ls $INPUT_DIR | wc -l)
sbatch --array=0-$((NUM_FILES - 1)) run_eden.slurm