#!/bin/bash
#SBATCH --job-name=yeast_250
#SBATCH --qos=cpu_qos
#SBATCH --partition=cpu
#SBATCH --nodes=1

#SBATCH --time=96:00:00
#SBATCH --output=/h/chchao/diff-yeast/generate/slurm_eval.out
#SBATCH --error=/h/chchao/diff-yeast/generate/slurm_eval.err
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