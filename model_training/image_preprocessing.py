#%%import packages
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
os.getcwd()

# %%
img = Image.open('c:\\Users\\Shuab.Akintola\\Desktop\\Pytorch_Ultimate\\Image\\stam.jpg')
img

#%%
img.size

# %%image preprocessing

transform = transforms.Compose([transforms.Resize(
    (300, 300)),
    transforms.RandomRotation(30),
    transforms.CenterCrop(150),
    transforms.RandomVerticalFlip(.4),
    transforms.Grayscale(),
    transforms.ToTensor(),   
    transforms.Normalize(0.4344, 0.1631)])
x = transform(img)
x

# %%
