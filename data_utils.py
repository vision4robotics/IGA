import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import cv2
import numpy as np


def img2tensor(img_arr):
    img_arr = img_arr.astype(np.float32)
    img_arr = img_arr.transpose(2, 0, 1)
    img_arr = img_arr[np.newaxis, :, :, :]
    init_tensor = torch.from_numpy(img_arr)
    return init_tensor
def normalize(im_tensor):
    im_tensor = im_tensor / 255.0
    im_tensor = im_tensor - 0.5
    im_tensor = im_tensor / 0.5
    return im_tensor
def tensor2img(tensor):
    tensor = tensor.squeeze(0).permute(1,2,0)
    img = tensor.cpu().numpy().clip(0,255).astype(np.uint8)
    return img


