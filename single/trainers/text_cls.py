import math
import os.path as osp
import re
import string
from typing import Optional
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import _transform, tokenize
from tqdm import tqdm
from pathlib import Path

from utils.model_utils import *
import torch.nn.functional as F
from trainers.model.text_encoder import TextEncoder
_tokenizer = _Tokenizer()
class CLIP_Zero_Shot_adapt(nn.Module):

    def __init__(self, model, classes, templates, device='cuda', dataset_name=None, log=None, txt_cls = None, cfg=None):
        super(CLIP_Zero_Shot_adapt, self).__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.device = device
        self.classes = classes
        self.model = model.to(device)
        self.log = log
        self.args = None
        self.txt_cls = txt_cls
        self.templates = templates
        self.txt_feas = self.txt_features(self.classes, self.templates)

        backbone_name = cfg.MODEL.BACKBONE.NAME
        self.if_vit=not backbone_name.startswith('R')
        
        self.backbone=backbone_name.replace("/", "_")    

        self.dim = self.model.text_projection.shape[1]

        print( self.model.text_projection.shape[1])
        

        self.txt_features_for_text_cls, self.labels_for_text_cls = self.txt_features_for_text_cls()
        
        self.text_encoder = TextEncoder(self.model)

       
        text_feat,text_label=self.txt_features_for_text_cls, self.labels_for_text_cls
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)   
        #new_feat=(Smask(text_feat,1.5))*text_feat
        head=self.get_avg(text_feat,text_label)
        self.text_adapter=nn.Linear(int(self.dim), len(classes),bias=False)
        self.one_hot_labels = F.one_hot(text_label, num_classes=max(text_label)+1)
        text_feat_ = text_feat / text_feat.norm(dim=-1, keepdim=True)  
        #cache=text_feat_.t()
  

        self.text_adapter.weight=nn.Parameter(text_feat_.cuda()) 
        self.drop=nn.Dropout(.2)
        #if text_feat.shape[0]>text_feat.shape[1]:
        #    mask=mask_Channel(text_feat,1.5)
        #else:
        mask=activate(text_feat,1.5)
        MBP=self.get_avg(mask,text_label)
        MBP=torch.abs(MBP)
        MBP=torch.exp(MBP)/torch.exp(MBP).sum(dim=0,keepdim=True)
        MBP=MBP/MBP.norm(dim=-1, keepdim=True)
        threshold =torch.quantile(MBP,cfg.rate)
        MBP[MBP>threshold]=1
        
        print(cfg.rate)
        print(MBP.sum(dim=-1).mean())
        self.MBP=MBP[text_label]
        #self.B=cal_B(text_feat,text_label,int(self.dim*cfg.rate)).cuda()





    def txt_features(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates]  # format with class
                texts = tokenize(texts).cuda()  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights


    @torch.no_grad()
    def get_avg(self,txt_feas,label):
        mean_text_feature = torch.zeros(len(self.classes), txt_feas.shape[-1]).to(self.device)
        count = torch.zeros(len(self.classes), 1).to(self.device)
        mean_text_feature = torch.scatter_add(mean_text_feature, dim=0, index=label.unsqueeze(1).repeat(1, txt_feas.shape[-1]), src=txt_feas)
        count = torch.bincount(label)

        mean_text_feature /= count.unsqueeze(1)

        return mean_text_feature
      
    def txt_features_for_text_cls(self):

        if self.txt_cls== '0':
            gpt3_prompts = None
            desc, labels_for_descriptions = gen_labels_with_classes(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == '1':
            gpt3_prompts = self.templates

            desc, labels_for_descriptions = gen_labels_with_templates(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == '2':
            # targetted_prompts
            path_to_file = f'./descriptions/tap/{self.dataset_name}.json'

            with open(path_to_file) as f:
                gpt3_prompts = json.load(f)

            desc, labels_for_descriptions = gen_labels_with_descrptions(self.classes, descriptions=gpt3_prompts)


        else:
            raise ValueError('Invalid txt_cls argument')

        Path(f'embeddings_icassp').mkdir(parents=True, exist_ok=True)

        if os.path.isfile(f'embeddings/{self.backbone}_{self.txt_cls}_{self.dataset_name}_embeddings.pt'):

            zeroshot_weights = torch.load(f'embeddings/{self.backbone}_{self.txt_cls}_{self.dataset_name}_embeddings.pt')
            print('******** Loaded Already Saved Embeddings *********')
            labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()

        else:
            print('******** No Embeddings Found --- Saving New Embeddings *********')

            labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()

            zeroshot_weights = []
            with torch.no_grad():
                for classname in tqdm(desc):
                    text = tokenize(classname).cuda()  # tokenize # (50, 77) --> 50 templates/texts from GPT
                    class_embeddings = self.model.encode_text(
                        text)  # embed with text encoder # (50, 512) --> embeddings for all 50 texts
                    #class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # L2 norm of the embeddings (dim 2)
                    zeroshot_weights.append(class_embeddings)
                zeroshot_weights = torch.stack(zeroshot_weights).cuda()  # (512, 10) --> 512 embeddings for 10 classes'
                torch.save(zeroshot_weights, f'embeddings/{self.backbone}_{self.txt_cls}_{self.dataset_name}_embeddings.pt')

        return zeroshot_weights.squeeze().float(), labels_for_descriptions
#torch.arange(len(self.classes)).to(self.device)
    
    def train_txt_clas(self, criteria,txt_feas=None,txt_label=None):
        if txt_feas==None or txt_label==None:
            txt_feas = self.txt_features_for_text_cls
            txt_label = self.labels_for_text_cls
        txt_feas = txt_feas / txt_feas.norm(dim=-1, keepdim=True)  
        #txt_feas=txt_feas-txt_feas.mean(dim=0)
        txt_feas=txt_feas+0.1*torch.randn_like(txt_feas).to(self.device)
        txt_feas=txt_feas*self.MBP
        #txt_feas=self.drop(txt_feas)
        feas=self.text_adapter(txt_feas)
        logit=criteria(feas, txt_label)
        loss =logit
        return loss
        
    def eval_text_adapter(self, x1):
        with torch.no_grad():
            txt_feas = self.txt_features_for_text_cls
            txt_feas = txt_feas / txt_feas.norm(dim=-1, keepdim=True)  
            txt_feas=txt_feas-txt_feas.mean(dim=0)
            img_features_1 = self.image_features(x1.float())
            feas=self.text_adapter(img_features_1)  #-img_features_1.mean(dim=0)
            return feas
             
    def eval_feat(self):
        with torch.no_grad():
            img_features_1 = self.test_feat["img"]
            feas=self.text_adapter(img_features_1)
            return feas    
        
        
    def image_features(self, images):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def forward(self, x1):
        with torch.no_grad():
            img_features = self.image_features(x1)
            out = img_features.float() @ self.txt_feas.float()
        return out
   
    @torch.no_grad()
    def cal_nosiy_mean(self):
        imgs=[]
        filename=f'embeddings/{self.backbone}_img.pt'
        if os.path.isfile(filename):
            print('==========loading img noise feature============')
            self.img_noise=torch.load(filename)
            return
        print('==========cal img noise feature============')
        for img in tqdm(self.noise_img_loader):
            imgs.append(self.image_features(img.float().cuda()))
        self.img_noise=torch.cat(imgs, dim=0)
        torch.save(self.img_noise,filename)
            
    @torch.no_grad()
    def cal_test_img_feat(self,dataloader):
      
        imgs={}
        imgs["img"]=[]
        imgs["label"]=[]
        self.test_feat={}
        filename=f'embeddings/img_{self.backbone}_{self.dataset_name}_seed{self.cfg.SEED}.pt'
        if os.path.isfile(filename):
            print('==========loading img test feature============')
            self.test_feat=torch.load(filename)
            return
        print('==========cal img test feature============')
        for inputs in tqdm(dataloader):
            img = inputs["img"]
            label = inputs["label"]
            imgs["img"].append(self.image_features(img.float().cuda()))
            imgs["label"].append(label)
        self.test_feat["img"]=torch.cat(imgs["img"], dim=0)
        self.test_feat["label"]=torch.cat(imgs["label"], dim=0)
        torch.save(self.test_feat,filename)
    @torch.no_grad()
    def eval_with_feat(self,i):
        labels = self.test_feat["label"].cuda()
        total=len(labels)
        correct_base = 0. 
        text_out = self.eval_feat()
        pred_base = torch.argmax(text_out, dim=1).cuda()
        correct_base=torch.eq(pred_base, labels).sum().item()
        top1 = (correct_base / total) * 100
        print(f'epoch{i},Accuracy: {top1}' )
        return top1


        
@TRAINER_REGISTRY.register()
class clip_adapt(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        #self.trans=_transform(clip_model.input_resolution.item())
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        print("Building ZERO-SHOT-MODEL CLIP")
        self.model = CLIP_Zero_Shot_adapt(model=clip_model, classes=classnames,
                                          templates=['a photo of a {}'], dataset_name = cfg.DATASET.NAME, txt_cls = cfg.txt_cls, cfg=cfg)
        self.register_model("adapt", self.model)
       
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        cfg = self.cfg
        test_trans=load_transforms(cfg)

        dm = DataManager(self.cfg, custom_tfm_test=test_trans, custom_tfm_train=test_trans)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def parse_batch_train(self, batch):

        if isinstance(batch, list):
            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(self.device)
        else:
            input = batch['img']
            input = input.to(self.device)

        label = batch["label"]
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


        