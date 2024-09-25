#!/bin/bash
#SBATCH --job-name=gen_6
#SBATCH -c 32
#SBATCH --qos=normal
#SBATCH --partition=rtx6000
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

#SBATCH --time=16:00:00
#SBATCH --output=/h/chchao/diff-yeast/slurm.out
#SBATCH --error=/h/chchao/diff-yeast/slurm.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
term_handler() {
	scontrol requeue "${SLURM_JOB_ID}"
}
trap term_handler SIGUSR1
source ~/.bashrc
conda activate cellpose
which python

cd /h/chchao/diff-yeast
/h/chchao/.conda/envs/cellpose/bin/python naive_video_gen.py config=test/test_6.ymal
/h/chchao/.conda/envs/cellpose/bin/python naive_video_gen.py config=test/test_7.ymal
/h/chchao/.conda/envs/cellpose/bin/python naive_video_gen.py config=test/test_8.ymal
/h/chchao/.conda/envs/cellpose/bin/python naive_video_gen.py config=test/test_9.ymal
/h/chchao/.conda/envs/cellpose/bin/python naive_video_gen.py config=test/test_10.ymal
child="$!"
wait "$child"