# Diffusion Model Generation for Yeast Cells -- Generate Videos

### GPU Request
```bash
srun -p a40 -q normal -c 16 --gres=gpu:1 --time 04:00:00 --pty bash
```

### Install Packages
```
pip install -r requirements.txt
pip install "opencv-python-headless<4.3"
pip install tensorflow
pip install torchvision
pip install imutils
pip install scikit-learn
pip install openpyxl
pip install cellpose
```


# Time
- selection: 31 s
- interpolation: 75 s

### Generate Video Examples
- (Testing) Generate a GIF with a specific structure:
```
python frame_matching.py config=test/test_1.yaml
python frame_interpolation.py config=test/test_1.yaml
python video_gt_gen_ray1 config=default.yaml
```

- Video ground truth generation
```
sh cmd/gen_orf_1.sh
sh cmd/gen_orf_2.sh
sh cmd/gen_orf_3.sh
sh cmd/gen_orf_4.sh
```

- Enhancement
```
python refine_sharpen.py
python refine_add_title.py
python refine_add_tfid.py
```

- ISSUE: cannot use A40 GPUs.
```
RuntimeError: Program 'ffmpeg' is not found; perhaps install ffmpeg using 'apt install ffmpeg'.
```

### Ray Generation

- Check the Resources
```
nproc --all
```

- CPU nodes
```
srun -p cpu -q cpu_qos -c 16 --nodes=1 --time 04:00:00 --pty bash
```

- Generation code
```
python ray_script/vector/video_gt_gen_ray_1.py config=default.yaml
```

# Bottleneck

- incomplete `R1` folder (cannot unzip `R1.zip`). need more space.