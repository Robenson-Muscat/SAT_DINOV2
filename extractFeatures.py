
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
import glob
from tqdm import tqdm
import torch.utils.data as data
import torch.nn as nn
import pytorch_lightning as pl
from utils_functions import alphanumeric_sort
import pandas as pd
import re
import torch
import dask.dataframe as dd
import sys
sys.path.append("HighResCanopyHeight/")
from models.backbone import SSLVisionTransformer



class InferenceDataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.filenames = sorted(glob.glob(os.path.join(path, '**/*.png'), recursive=True) + \
                                glob.glob(os.path.join(path, '**/*.jpg'), recursive=True) + \
                                glob.glob(os.path.join(path, '**/*.JPG'), recursive=True) + \
                                glob.glob(os.path.join(path, '**/*.PNG'), recursive=True) + \
                                glob.glob(os.path.join(path, '**/*.jpeg'), recursive=True) + \
                                glob.glob(os.path.join(path, '**/*.JPEG'), recursive=True) + \
                                glob.glob(os.path.join(path, '**/*.tif'), recursive=True))

        self.filenames = sorted(self.filenames, key=alphanumeric_sort)

        self.transform = transform

    def __getitem__(self, index):
        img_path = self.filenames[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, img_path

    def __len__(self):
        return len(self.filenames)


class SSLembed(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SSLVisionTransformer(out_indices=(9, 16, 22, 29), embed_dim=1280,
                                             num_heads=20, depth=32, pretrained=None)

    def forward(self, x):
        x = self.backbone(x)
        return x


class SSLModule(pl.LightningModule):
    def __init__(self,
                 device,
                 ssl_path="HighResCanopyHeight/saved_checkpoints/compressed_SSLhuge.pth"):
        super().__init__()
        self.device_type = device.type
        self.chm_module_ = SSLembed().to(device).eval()

        # Charger le checkpoint
        ckpt = torch.load(ssl_path, map_location=device)

        
        if self.device_type == "cpu":
            self.chm_module_ = torch.quantization.quantize_dynamic(
                self.chm_module_,
                {torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d},
                dtype=torch.qint8)

        self.chm_module_.load_state_dict(ckpt, strict=False)
        self.chm_module = lambda x: 10 * self.chm_module_(x)

    def forward(self, x):
        x = self.chm_module(x)
        return x


def predict_and_save(model, dl, device, save_dir, batch_size):
    os.makedirs(save_dir, exist_ok=True)
    filenames, feature_map_avgs = [], []
    count = 0
    for i, batch in tqdm(enumerate(dl), desc='Processing batches'):
        images, files = batch[0].to(device), batch[1]

        with torch.no_grad():
            pred = model(images)
            pred = pred[0][0][0].cpu().numpy().reshape([1280, 14 * 14]).transpose()

            feature_map_avgs.append(pred)
            filenames += list(files)

        #Store features in CSV files
            

        count += 1
        if count % 300 == 0:
            feature_map_avgs = np.array(feature_map_avgs).reshape(-1, 1280)
            df = pd.DataFrame(data=feature_map_avgs)
            df.to_csv(os.path.join(save_dir, f'embeddings_s{count // 300}.csv'), index=False)
            filenames, feature_map_avgs = [], []

    feature_map_avgs = np.array(feature_map_avgs).reshape(-1, 1280)
    df = pd.DataFrame(data=feature_map_avgs)
    df.to_csv(os.path.join(save_dir, f'embeddings_s{count // 300 + 1}.csv'), index=False)


def getFeatures(path, save_dir, device):
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = SSLModule(device).to(device)
    model.eval()

  
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.420, 0.411, 0.296), (0.213, 0.156, 0.143))
    ])

  
    batch_size = 1
    dataset = InferenceDataset(path, transform)
    dl = data.DataLoader(dataset, batch_size=batch_size)


    predict_and_save(model, dl, device, save_dir, batch_size)