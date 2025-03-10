import cv2
import gradio as gr
import os
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import gdown
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.system("git clone https://github.com/xuebinqin/DIS")
os.system("mv DIS/IS-Net/* .")

# 要在当前文件夹执行一下，否则用comfyui运行会直接下载到comfyui顶层文件夹