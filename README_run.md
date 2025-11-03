
```bash
srun -A aip-rahulgk --gres=gpu:h100:1 --cpus-per-task=8 --mem=32G --time=3:00:00 --pty bash
srun -A aip-rahulgk --gres=gpu:l40s:1 --cpus-per-task=8 --mem=32G --time=12:00:00 --pty bash
srun -A aip-rahulgk --gres=gpu:l40s:1 --cpus-per-task=8 --mem=32G --time=3:00:00 --pty bash
srun --pty --overlap --jobid 418011 bash

conda deactivate
uv venv --python=3.11 /home/chchao0/projects/aip-rahulgk/chchao0/venvs/diff-yeast
source /home/chchao0/projects/aip-rahulgk/chchao0/venvs/diff-yeast/bin/activate
uv pip install -r /home/chchao0/projects/aip-rahulgk/chchao0/vq-test/requirements.txt

uv pip install -r requirements.txt
uv pip install -r requirements_all.txt
uv pip install pillow numpy
uv pip install pandas
uv pip install matplotlib
uv pip install scikit-learn
```

### Extract
```bash
python extract_channels.py --prefix /home/chchao0/projects/aip-rahulgk/chchao0/generated_video --name YAL001C --method nucleus --out extracted
python extract_channels_red.py --input-dir /home/chchao0/projects/aip-rahulgk/chchao0/generated_video --names YAL001C --methods nucleus --output-dir extracted_red
python extract_channels_green.py --input-dir /home/chchao0/projects/aip-rahulgk/chchao0/generated_video --names YAL001C --methods nucleus --output-dir extracted_green
python extract_channels_black.py --input-dir /home/chchao0/projects/aip-rahulgk/chchao0/generated_video --names YAL001C --methods nucleus --output-dir extracted_black
```


### TSNE Plots
```bash
python cluster_plot.py
```
