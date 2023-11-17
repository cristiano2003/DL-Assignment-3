import pandas as pd
from mask2csv import mask2string
import os 
import sys
import numpy as np
from PIL import Image
import cv2 as cv
import time
import imageio
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision.transforms import *
from collections import OrderedDict
from UnetData import UNetTestDataClass
import segmentation_models_pytorch as smp
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = smp.UnetPlusPlus(
    encoder_name='resnet50', 
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=3     
)

checkpoint = torch.load('/kaggle/working/unet_model.pth')

new_state_dict = OrderedDict()
for k, v in checkpoint['model'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)



transform = Compose([Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                     PILToTensor()])


test_path = '/kaggle/input/bkai-igh-neopolyp/test/test/'
unet_test_dataset = UNetTestDataClass(test_path, transform)
test_dataloader = DataLoader(unet_test_dataset, batch_size=8, shuffle=True)


model.eval()
if not os.path.isdir("/kaggle/working/predicted_masks"):
    os.mkdir("/kaggle/working/predicted_masks")
for _, (img, path, H, W) in enumerate(test_dataloader):
    a = path
    b = img
    h = H
    w = W
    
    with torch.no_grad():
        predicted_mask = model(b)
    for i in range(len(a)):
        image_id = a[i].split('/')[-1].split('.')[0]
        filename = image_id + ".png"
        mask2img = Resize((h[i].item(), w[i].item()), interpolation=InterpolationMode.NEAREST)(ToPILImage()(F.one_hot(torch.argmax(predicted_mask[i], 0)).permute(2, 0, 1).float()))
        mask2img.save(os.path.join("/kaggle/working/predicted_masks/", filename))


MASK_DIR_PATH = '/kaggle/working/predicted_masks' 
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(r'output.csv', index=False)