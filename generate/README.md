# Diffusion Model Generation for Yeast Cells -- Generate Videos

### Install Packages
```
pip install -r requirements.txt
pip install "opencv-python-headless<4.3"
pip install tensorflow
pip install torchvision
pip install imutils
pip install scikit-learn
pip install openpyxl
```

### Generate Video Examples
- (Testing) Generate a GIF with a specific structure:
```
python video_gt_gen.py config=test/test_1.yaml
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
