
import torch
import torchvision.transforms as T
import os
import h5py
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from dassl.utils import read_image
class ImgFeatDataset(TorchDataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = h5py.File(file_path, 'r')
        self.img = self.file['img']

        self.labels = self.file['label']   
    def __len__(self):
       
        return len(self.img)

    def __getitem__(self, idx):

        output = {
            "img": torch.tensor(self.img [idx]),
            "label":torch.tensor(self.labels[idx]),
        }
        return output
    
class textDataset(TorchDataset):
    def __init__(self, caption,label):
        self.caption=caption
        self.label =label
    def __len__(self):
       
        return len(self.label)

    def __getitem__(self, idx):

        output = {
            "caption": self.caption [idx],
            "label":torch.tensor(self.label[idx]),
        }
        return output
        