import torch
from utils import Unet3D, GaussianDiffusion, Evaluator

num_frames = 193
split = 8

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

# model = torch.nn.DataParallel(model)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    num_frames = num_frames // split,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

diffusion = diffusion.to('cuda')

evaluator = Evaluator(diffusion)

evaluator.load('results/model-1.pt')
evaluator.eval()