import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers import DDIMScheduler, DDPMPipeline
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
import os
from data.dataset import data_loader


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



root_dir = "data/TB_data"
loader_, dataset = data_loader(root_dir=root_dir, batch_size=12)
train_dataloader = loader_['train']

def train(train_dataloader, epoch_st, epoch_end, lr=1e-4):

    image_pipe = DDPMPipeline.from_pretrained("saved_model/my-finetuned-model_69")
    image_pipe.to(device);
    grad_accumulation_steps = 2  # @param

    optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=lr)
    

    losses = []

    for epoch in range(epoch_st,epoch_end):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            images, labels = batch
            clean_images = images.to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                image_pipe.scheduler.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction for the noise
            noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

            # Compare the prediction with the actual noise:
            loss = F.mse_loss(
                noise_pred, noise
            )  # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)

            # Store for later plotting
            losses.append(loss.item())

            # Update the model parameters with the optimizer based on this loss
            loss.backward(loss)

            # Gradient accumulation:
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        print(
            f"Epoch {epoch} average loss: {sum(losses[-len(train_dataloader):])/len(train_dataloader)}"
        )
        image_pipe.save_pretrained(f"saved_model/my-finetuned-model_{epoch}")

    return image_pipe


model = train(train_dataloader=train_dataloader,epoch_st=70,epoch_end=100)