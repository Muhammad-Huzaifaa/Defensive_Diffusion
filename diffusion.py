import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from data.dataset import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "data/TB_data"
loader_, dataset = data_loader(root_dir=root_dir, batch_size=12)
train_dataloader = loader_['train']