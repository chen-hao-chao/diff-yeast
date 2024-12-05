import torch
from utils import Unet3D, GaussianDiffusion, Trainer

num_frames = 192
split = 16 #8
image_size = 80

model = Unet3D(
    dim = image_size,
    dim_mults = (1, 2, 4, 8)
)

model = torch.nn.DataParallel(model)

diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    num_frames = num_frames // split,
    timesteps = 10,#00,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

diffusion = diffusion.to('cuda')

trainer = Trainer(
    diffusion,
    '../gt_videos',                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = 32,
    train_lr = 1e-4,
    save_and_sample_every = 1,#000,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    split = split,
    results_folder = './results_test_2'
)

trainer.train()