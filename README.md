# Diffusion Model Generation for Yeast Cells

### Create Conda Environment
```bash=
conda create --name diffy python=3.8
conda activate diffy
```

### GPU request
```
salloc --time=01:30:0 --mem-per-cpu=4G --ntasks=1 --account=def-rahulgk
```

### Virtual Env
- Load opencv before installing `cellpose`
```
module load gcc
module load opencv/4.11.0
module load python/3.10
module load arrow
source yeast_env/bin/activate
```

### Ceder Installation
- conda installation
```
pip install -r requirement_all.txt
```
- virtualenv installation
```
pip install -r requirements_rest.txt
tifffile
scikit-image
tensorflow_hub
mediapy
hydra-core
ray
ray[tune]
colorama
```



### UI
```
streamlit run test_ui_workable.py
```