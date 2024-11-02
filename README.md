# Diffusion Model Generation for Yeast Cells

### Create Conda Environment
```bash=
conda create --name cellpose2 python=3.8
conda activate cellpose2
```

### Install Packages
```
pip install -r requirements.txt
pip install "opencv-python-headless<4.3"
pip install tensorflow
pip install torchvision
pip install imutils
pip install scikit-learn
```

### Generate Video Examples
- Generate GIFs:
```
sh cmd/gen_1.sh
sh cmd/gen_2.sh
```

- (Testing) Generate a GIF with a specific structure:
```
python video_gt_gen.py config=test/test_1.yaml
```

- ISSUE: cannot use A40 GPUs.
```
RuntimeError: Program 'ffmpeg' is not found; perhaps install ffmpeg using 'apt install ffmpeg'.
```

- Video ground truth generation
```
sh cmd/gen_orf_1.sh
```