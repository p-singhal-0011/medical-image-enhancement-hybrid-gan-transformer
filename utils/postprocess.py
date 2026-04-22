import torch
from torchvision.utils import save_image

def save_output(tensor, path):
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    save_image(tensor, path)
