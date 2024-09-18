"""
Goal: To experiment with different diffusion models
    Mini Goal: Get a diffusion model to be great at pixel art
        Steps: Define my own network, train, test, repeat
    Mini Goal: Modify Training loop to create colored flip illusion
        Steps: Define my own network, train, test, repeat
"""

import torch

from dataset import dataset
from training_config import config

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.train_batch_size, shuffle=True
)

import torch
from diffusers import DDPMScheduler
from PIL import Image

from model import model

sample_image = dataset[0]["images"].unsqueeze(0)
print("Input shape:", sample_image.shape)
print("Output shape:", model(sample_image, timestep=0).sample.shape)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

img = Image.fromarray(
    ((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0]
)

from diffusers.optimization import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

from accelerate import notebook_launcher

from train import train_loop

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)
model.save_pretrained("pixel-diffusion")
