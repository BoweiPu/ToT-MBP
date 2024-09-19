import torch
import json
import torchvision.transforms as transforms
import PIL.Image as Image
from PIL import ImageFilter
import random
from clip import clip 
import os
import open_clip
import numpy as np



def activate(feas,rate):
    mask = torch.zeros_like(feas)
    feas=feas-feas.mean()
    condition =  (feas > rate * feas.std()) 
    mask[condition]+=1
    condition =  (feas <- rate * feas.std()) 
    mask[condition]-=1
    #mask[mask!=0]=1
    return mask



def cal_B(feas,label,num):
    mask = torch.zeros(max(label)+1,feas.shape[1]).cuda()
    select={}
    for f,l in zip(feas,label):
        l=l.item()
        if l not in select.keys():
            select[l]=[]
        select[l].append(f)
    for key in select.keys():
        count=0
        for i ,fi in enumerate(select[key]):
            for j ,fj in enumerate(select[key]):
                if i!=j and i>j:
                    mask[key]+=fi*fj
                    count=1+count
        mask[key]=mask[key]/count
    out=feas.var(dim=0) 
    mask=mask.mean(dim=0)          
    re=out*0.3-mask*.7
    _, indices = torch.topk(re, num, largest=True)
    B=torch.zeros((feas.shape[1]))
    B[indices]=1

    return B

import torch
import torch.nn.functional as F
def get_avg(txt_feas,label):
        numcls=max(label)+1
        mean_text_feature = torch.zeros(numcls, txt_feas.shape[-1])
        count = torch.zeros(numcls, 1)
        mean_text_feature = torch.scatter_add(mean_text_feature, dim=0, index=label.unsqueeze(1).repeat(1, txt_feas.shape[-1]), src=txt_feas)
        count = torch.bincount(label)
        mean_text_feature /= count.unsqueeze(1)
        return mean_text_feature 
def InfoNCE_Loss(similarity_matrix, mask_matrix, temperature=0):
    """
    InfoNCE Loss implementation using PyTorch.
    Args:
        similarity_matrix (torch.Tensor): A similarity matrix.
        mask_matrix (torch.Tensor): A mask matrix where 1s indicate positive samples and 0s indicate negative samples. 
        temperature (float): A temperature parameter for the softmax function. softmax
    Return:
        loss (torch.Tensor): Computed InfoNCE loss. 计算的InfoNCE损失
    """
    mask_matrix = mask_matrix.type(torch.bool)
    similarity_matrix=similarity_matrix / temperature
    similarity_matrix=similarity_matrix-torch.max(similarity_matrix,dim=-1, keepdim=True)[0]
    pos_sim = similarity_matrix[mask_matrix].view(similarity_matrix.size(0), -1)   
    log_sum_exp_pos = torch.log(torch.sum(torch.exp(pos_sim), dim=1))
    log_sum_exp_all = torch.log(torch.sum(torch.exp(similarity_matrix), dim=1))
    loss = -torch.mean(log_sum_exp_pos - log_sum_exp_all)
    return loss
def InfoNCE_Loss000(similarity_matrix, mask_matrix, temperature=0):
    """
    InfoNCE Loss implementation using PyTorch.
    Args:
        similarity_matrix (torch.Tensor): A similarity matrix.
        mask_matrix (torch.Tensor): A mask matrix where 1s indicate positive samples and 0s indicate negative samples. 
        temperature (float): A temperature parameter for the softmax function. softmax
    Return:
        loss (torch.Tensor): Computed InfoNCE loss. 计算的InfoNCE损失
    """
    mask_matrix = mask_matrix.type(torch.bool)
    
    pos_sim = similarity_matrix[mask_matrix].view(similarity_matrix.size(0), -1)
    neg_sim = similarity_matrix[~mask_matrix].view(similarity_matrix.size(0), -1)
    
    pos_sim = pos_sim / temperature
    neg_sim = neg_sim / temperature
    
    max_val = torch.max(pos_sim, dim=1, keepdim=True)[0]
    
    log_sum_exp_pos = pos_sim - max_val - torch.log(torch.sum(torch.exp(pos_sim - max_val), dim=-1, keepdim=True))
    log_sum_exp_neg = neg_sim - max_val - torch.log(torch.sum(torch.exp(neg_sim - max_val), dim=-1, keepdim=True))
    
    loss = -torch.mean(log_sum_exp_pos) + torch.mean(torch.log(torch.sum(torch.exp(log_sum_exp_neg), dim=-1)))
    
    return loss
def update_teacher_params(teacher_model, student_model, alpha):
    """
    更新教师模型参数的函数。
    教师参数 = α * 教师参数 + (1 - α) * 学生参数
    """
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=(1 - alpha))
            
def custom_cosine_embedding_loss(image_vectors, text_vectors, match_matrix, margin):
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(image_vectors.unsqueeze(1), text_vectors.unsqueeze(0), dim=-1)

    # 创建匹配标签的张量
    match_labels = torch.where(match_matrix == 1, torch.tensor(1), torch.tensor(-1))

    # 计算对比损失
    positive_loss = torch.max(torch.tensor(0.), margin - cos_sim)  # 匹配样本的损失
    negative_loss = torch.max(torch.tensor(0.), cos_sim - margin)  # 不匹配样本的损失

    # 计算总的对比损失
    loss = (match_labels * positive_loss + (1 - match_labels) * negative_loss).mean()

    return loss
def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root='all_weights')
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model
def load_transforms(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    pretained= cfg.MODEL.BACKBONE.PRETRAINED
    _, _, preprocess = open_clip.create_model_and_transforms(backbone_name, pretrained=pretained,cache_dir="weight")
    return preprocess

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, random_transform):
        self.base_transform = base_transform  # from clip
        self.random_tranform = random_transform  # random transforms (currently from simsiam)

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.random_tranform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_random_transform(ndim):
    normalize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                         (0.26862954, 0.26130258, 0.27577711))])

    blur = GaussianBlur()

    return transforms.Compose([
        transforms.RandomResizedCrop(ndim, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        blur,
        normalize
    ])


te_transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    lambda image: image.convert("RGB"),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

transform_default_clip_weakly_aug = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    lambda image: image.convert("RGB"),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

tr_transforms = TwoCropsTransform(te_transform, get_random_transform(224))

def gen_labels_with_templates(classes, descriptions):
    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        if '_' in classname:
            classname = classname.replace('_', ' ')
        print(classname)
        for descp in descriptions:
            descp = descp.format(classname)
            desc_.append(descp)
            labels.append(i)
    return desc_, labels


def gen_labels_with_captions(classes, folder_path, args):
    if args.dataset == 'imagenet':
        desc_ = []
        labels = []
        cls_name_dict = {}
        with open(os.path.join(folder_path, 'imagenet_class_index.json')) as f:
            cls_idx = json.load(f)
        for k, v in cls_idx.items():
            cls_name_dict[v[0]] = v[1]
        for i, classname in enumerate(cls_name_dict.keys()):
            with open(os.path.join(folder_path, f'{classname}.txt'), 'r') as f:
                for line in f:
                    desc_.append(line.split(" ", 1)[1].replace("\n", "").lower())
                    labels.append(i)

        return desc_, labels

    desc_ = []
    labels = []
    classes_to_care = ['aquarium fish', 'lawn mower', 'maple tree', 'oak tree', 'pickup truck', 'pine tree',
                       'sweet pepper', 'willow tree']
    for i, classname in enumerate(classes):
        if classname in classes_to_care:
            split_ = 2
        else:
            split_ = 1
        with open(os.path.join(folder_path, f'{classname}.txt'), 'r') as f:
            for line in f:
                desc_.append(line.split(" ", split_)[split_].replace("\n", "").lower())
                labels.append(i)
    return desc_, labels


def gen_labels_with_captions_blip_2(classes, folder_path, args):
    with open(folder_path) as f:
        lines = f.readlines()
    desc_ = []
    labels = []

    classes_to_care = ['aquarium fish', 'lawn mower', 'maple tree', 'oak tree', 'pickup truck', 'pine tree',
                       'sweet pepper', 'willow tree']

    for i, c in enumerate(classes):
        if c in classes_to_care:
            split_ = 2
        else:
            split_ = 1
        for l in lines:
            if l.strip().split(' ')[0].split('/')[-2] == c:
                labels.append(i)
                desc_.append(l.strip().split(' ', split_)[split_].replace("\n", "").lower())
                print(l.strip().split(' ', split_)[split_].replace("\n", "").lower())
    return desc_, labels


def gen_labels_with_classes(classes, descriptions):
    # for direct class -> class
    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        desc_.append(classname)
        labels.append(i)
    return desc_, labels


def gen_labels_with_classes_and_simple_template(classes, descriptions):
    # for direct class -> simple template
    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        descp = f'a photo of a {classname}'
        desc_.append(descp)
        labels.append(i)
    return desc_, labels


def gen_labels_with_synonyms(classes, folder_path, args):
    with open(os.path.join(folder_path, f'{args.dataset}_cleaned.json')) as f:
        cls_idx = json.load(f)
    desc_ = []
    labels = []
    for i, (k, v) in enumerate(cls_idx.items()):
        classes = cls_idx[k].split(',')
        for j, classname in enumerate(classes):
            desc_.append('a photo of a ' + classname + '.')
            labels.append(i)
    return desc_, labels


def gen_labels_with_descrptions(classes, descriptions):
    desc_ = []
    labels = []
    classes = descriptions.keys() # uncomment this for sun397
    for i, classname in enumerate(classes):
        for desc in descriptions[classname]:
            desc_.append(desc)
            labels.append(i)
    return desc_, labels


def gen_labels_with_expanded_labels_imagenet(folder, args):
    if args.dataset != 'imagenet':
        raise ValueError('Only for imagenet')
    with open(os.path.join(folder, 'imagenet_expanded_labels.txt')) as f:
        data = f.readlines()
    exp_cls = list()
    for i, d in enumerate(data):
        exp_cls.append(d.split(' ', 1)[1].split(' ', 1)[1].replace('\n', '').replace('\'', '').replace('\"', ''))
    desc_ = []
    labels = []
    for i, c in enumerate(exp_cls):
        cls = c.split(',')
        for j, c_ in enumerate(cls):
            if c_ == '':
                continue
            desc_.append('a photo of a ' + c_.replace(' ', ''))
            labels.append(i)
    return desc_, labels

def gen_labels_with_descrptions_and_clsname(classes, descriptions):
    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        for desc in descriptions[classname]:
            desc_.append(classname + ': ' + desc)
            labels.append(i)

    return desc_, labels