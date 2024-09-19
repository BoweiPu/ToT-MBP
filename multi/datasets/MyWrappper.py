import torch
import torchvision.transforms as T
import os
import h5py
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from dassl.utils import read_image

INTERPOLATION_MODES = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "nearest": Image.NEAREST,
}


def split_image(image, num_splits):
    w, h = image.size
    w_step, h_step = w // num_splits, h // num_splits
    return [image.crop((min(i * w_step, w - w_step), min(j * h_step, h - h_step), min((i + 1) * w_step, w), min((j + 1) * h_step, h))) for i in range(num_splits) for j in range(num_splits)]


class Split_DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        self.k_tfm=1

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }
        
        img0 = read_image(item.impath)
        images_3x3 = split_image(img0, 3)
        images_2x2 = split_image(img0, 2)
        #print(images_3x3)
        
        output["img"] = self.transform(img0)
        output['splits_3']= self.transform_imageList(self.transform, images_3x3)
        output['splits_2']= self.transform_imageList(self.transform, images_2x2)
     

        return output

    def transform_imageList(self, tfm, imgList):#[9 ,B,3,224,224]
        output=[]
        for img in imgList:
            output.append(tfm(img))
        return torch.stack(output,dim=0)




class TextFeatureDataset(TorchDataset):
    def __init__(self, directory):
        self.directory = directory
        self.file_paths = self._get_file_paths()

    def __len__(self):
        # 这里返回总的样本数量，需要按照您的文件存储方式进行计算
        total_count = 0
        for file_path in self.file_paths:
            data = torch.load(os.path.join(self.directory,f"{file_path}.pt"))
            total_count += len(data)
        return total_count

    def _get_file_paths(self):
        # 获取所有的.pt文件名，并去除后缀
        return sorted([int(os.path.splitext(f)[0]) for f in os.listdir(self.directory) if f.endswith('.pt')])

    def __getitem__(self, idx):
        # 查找大于idx的最小文件名
        next_files = [f for f in self.file_paths if f > idx]

        if not next_files:
            raise IndexError(f"No file found for index {idx},rebuild features please")

        next_file_idx = min(next_files)

        # 查找小于idx的最大文件名，如果不存在，则使用idx本身
        prev_file_idx = max([f for f in self.file_paths if f <= idx], default=idx)

        # 计算在文件内部的索引
        internal_idx = idx - prev_file_idx

        # 加载数据
        file_name = f"{next_file_idx}.pt"
        file_path = os.path.join(self.directory, file_name)
        data = torch.load(file_path)
        if not internal_idx < len(data):
            print(f'error:{idx},prev_file_idx:{prev_file_idx}')
        data=data[internal_idx]
        
        output = {
            "feature": data[0],
            "caption": data[1],
            "label":data[2],
        }
        return output

class TextFeatureDataset(TorchDataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = h5py.File(file_path, 'r')
        self.feature = self.file['feature']
        self.caption = self.file['caption']
        self.labels = self.file['label']      
        
    def __len__(self):
       
        return len(self.feature)

    def __getitem__(self, idx):

        output = {
            "feature": torch.tensor(self.feature [idx]),
            "caption": torch.tensor(self.caption[idx]),
            "label":torch.tensor(self.labels[idx]),
        }
        return output
class ImgFeatDataset(TorchDataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = h5py.File(file_path, 'r')
        self.img = self.file['img']
        self.img_2 = self.file['img_2']
        self.img_3 = self.file['img_3']
        self.labels = self.file['label']   
        self.feat = self.file['feat']   
        
    def __len__(self):
       
        return len(self.img)

    def __getitem__(self, idx):

        output = {
            "img": torch.tensor(self.img [idx]),
            "img2": torch.tensor(self.img_2[idx]),
            "img3":torch.tensor(self.img_3[idx]),
            "label":torch.tensor(self.labels[idx]),
            "feat":torch.tensor(self.feat[idx]),
        }
        return output
    

class ImgFeatureDataset(TorchDataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = h5py.File(file_path, 'r')
        self.img = self.file['img']
        self.img_2 = self.file['img_2']
        self.img_3 = self.file['img_3']
        self.labels = self.file['label']   
     
        
    def __len__(self):
       
        return len(self.img)

    def __getitem__(self, idx):

        output = {
            "img": torch.tensor(self.img [idx]),
            "img2": torch.tensor(self.img_2[idx]),
            "img3":torch.tensor(self.img_3[idx]),
            "label":torch.tensor(self.labels[idx]),
          
        }
        return output
    


class GenNoisyImgData(TorchDataset):

    def __init__(self, transform=None):
        self.transform = transform  # accept list (tuple) as input

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        item = self.data_source[idx]
        mean = [-0.0471, -0.0314,  0.0139]
        std = [1.0223, 1.0231, 1.0319]

        # 图像尺寸，例如 256x256
        H, W = 224,224

        # 为每个通道生成噪声
        noise_channel1 = torch.normal(mean=mean[0], std=std[0], size=(H, W))
        noise_channel2 = torch.normal(mean=mean[1], std=std[1], size=(H, W))
        noise_channel3 = torch.normal(mean=mean[2], std=std[2], size=(H, W))

        # 将三个通道的噪声堆叠起来
        img0 = torch.stack([noise_channel1, noise_channel2, noise_channel3], dim=0)

        output = { }
        
        img0 = read_image(item.impath)
        images_3x3 = split_image(img0, 3)
        images_2x2 = split_image(img0, 2)
        #print(images_3x3)
        
        output["img"] = img0
        output['splits_3']= self.transform_imageList(self.transform, images_3x3)
        output['splits_2']= self.transform_imageList(self.transform, images_2x2)
     

        return output

    def transform_imageList(self,  imgList):#[9 ,B,3,224,224]
        output=[]
        for img in imgList:
            output.append(img)
        return torch.stack(output,dim=0)
