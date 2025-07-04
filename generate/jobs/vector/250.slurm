#!/bin/bash
#SBATCH --job-name=yeast_all
#SBATCH -c 32
#SBATCH --qos=normal
#SBATCH --partition=t4v2
#SBATCH --gres=gpu:1

#SBATCH --time=16:00:00
#SBATCH --output=/h/chchao/diff-yeast/generate/slurm_eval1.out
#SBATCH --error=/h/chchao/diff-yeast/generate/slurm_eval1.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

term_handler() {
	scontrol requeue "${SLURM_JOB_ID}"
}
trap term_handler SIGUSR1
source ~/.bashrc
conda activate cellpose
which python

python video_gt_gen_ray1.py config=default.yaml
child="$!"
wait "$child"