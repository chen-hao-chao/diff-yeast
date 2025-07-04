#!/bin/bash
#SBATCH --job-name=yeast_fi_2
#SBATCH -c 40
#SBATCH --qos=normal
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1

#SBATCH --time=16:00:00
#SBATCH --output=/h/chchao/diff-yeast/generate/slurm_eval_fi_2.out
#SBATCH --error=/h/chchao/diff-yeast/generate/slurm_eval_fi_2.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

term_handler() {
	scontrol requeue "${SLURM_JOB_ID}"
}
trap term_handler SIGUSR1
source ~/.bashrc
conda activate cellpose
which python

python video_gt_gen_2_fi.py config=default.yaml
child="$!"
wait "$child"