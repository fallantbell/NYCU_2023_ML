import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image

import json


class iclevr_dataset(Dataset):
    def __init__(self, args):
        self.root = args.data_root
        
        self.transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        self.mode = args.mode

        self.imgname = []
        self.one_hot = []
        with open('objects.json', 'r') as f:
            O = json.load(f)

        if args.mode == "train":
            with open('train.json', 'r') as f:
                J = json.load(f)
            for key in J.keys():
                self.imgname.append(self.root+"/"+key)
                oh = np.zeros(24)
                objs = J[key]
                for obj in objs:
                    oh[O[obj]] = 1

                self.one_hot.append(oh)    
        elif args.mode == "test":
            with open('test.json', 'r') as f:
                J = json.load(f)
            for objs in J:
                oh = np.zeros(24)
                for obj in objs:
                    oh[O[obj]] = 1

                self.one_hot.append(oh)    
        else:
            with open('new_test.json', 'r') as f:
                J = json.load(f)
            for objs in J:
                oh = np.zeros(24)
                for obj in objs:
                    oh[O[obj]] = 1

                self.one_hot.append(oh)    
            
    def __len__(self):
        return len(self.one_hot)

    def __getitem__(self, index):
        if self.mode == "train":
            target_image = Image.open(self.imgname[index]).convert('RGB')
            target_image = self.transform(target_image)
        else:
            target_image = self.one_hot[index]

        condition = self.one_hot[index]

        return target_image, condition


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path):
    grid = torchvision.utils.make_grid(images)
    save_image(grid,path)

# def save_images(images, path, **kwargs):
#     grid = torchvision.utils.make_grid(images, **kwargs)
#     ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
#     im = Image.fromarray(ndarr)
#     im.save(path)


def get_data(args):
    print(f"args {args}")
    
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("test", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("test", run_name), exist_ok=True)