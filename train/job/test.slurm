#!/bin/bash
#SBATCH --job-name=results_test
#SBATCH -c 16 
#SBATCH --qos=a100_chchao
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:4

#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm.out
#SBATCH --error=logs/slurm.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
term_handler() {
	scontrol requeue "${SLURM_JOB_ID}"
}
trap term_handler SIGUSR1
source ~/.bashrc
conda activate cellpose
which python

python train.py
child="$!"
wait "$child"