import os.path as osp
from dassl.data.data_manager import DataManager
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.models._utils import IntermediateLayerGetter
import shutil
from tqdm import tqdm
import pickle5 as pickle
from timm.models.layers import trunc_normal_
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import os
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from datasets.MyWrappper import ImgFeatDataset, ImgFeatureDataset, Split_DatasetWrapper, TextFeatureDataset
from trainers.model.groupBlock import QueryBlock

from ..utils import *
_tokenizer = _Tokenizer()
import pdb
from torch.nn.parameter import Parameter
from ..imagenet_templates import *
def scale_tensor_to_target(tensor, target_modal, original_modal,noise):
    

    mean = torch.mean(tensor)
    std_target=torch.std(target_modal)
    std_original=torch.std(original_modal)

    alpha = std_target /std_original
    
    scaled_tensor =alpha*(tensor-mean) +mean+(std_original*2)*noise

    return scaled_tensor


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "COCO": "a photo of a {}."
}

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts, if_embedding=True, if_sequence=False,gauss=None):
        if not if_embedding:
            tokenized_prompts = prompts
            prompts = self.token_embedding(prompts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = prompts + self.positional_embedding.type(self.dtype)
        if gauss is not None:
            # 在序列维度上拼接captions和gaussian_tokens
            x = torch.cat([ x[:,:72,:],gauss], dim=1)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        
        if if_sequence:
            x = x @ self.text_projection  # NLD * Dd = NLd
            return x
        else:
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # ND * Dd = Nd
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
            return x




class DenseCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, return_interm_layers=False):
        super().__init__()
      

        self.classnames = classnames
        self.text_encoder = TextEncoder(clip_model)

        self.model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.drop_rate = 0.0

        self.img_feature_mean=0
        self.text_feature_mean=0
        self.img_feat_mean=0
        self.text_feat_mean=0
        self.img_feat_shape=[]
        self.drop = nn.Dropout(.1)
        
        #self.groupBlock.half()
        backbone_name = cfg.MODEL.BACKBONE.NAME
        self.if_vit=not backbone_name.startswith('R')
        self.backbone=backbone_name.replace('/','-')
        self.dim=self.model.text_projection.shape[1]
        self.head = nn.Linear(self.dim, len(classnames), bias=False)
        self.local_head= nn.Linear(self.dim,len(classnames), bias=False)
        self.QueryBlock=QueryBlock(dtype=self.dtype,num_token=16,dim=self.dim,low_dim=len(classnames))
        self.filter_local=None
        self.filter_head=None
        print(f'if_vit:{self.if_vit}')
        self.temperature = torch.tensor(3.0, dtype=clip_model.dtype)  #  exp(3.91) = 50
        if not self.if_vit :
            self.return_interm_layers = return_interm_layers
            if return_interm_layers:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {"layer4": "0"}
            self.visual_encoder = IntermediateLayerGetter(self.model.visual, return_layers)
            self.positional_embedding = self.model.visual.attnpool.positional_embedding[1::]
            self.v_linear_weight = self.model.visual.attnpool.v_proj.weight
            self.v_linear_bias = self.model.visual.attnpool.v_proj.bias
            self.c_linear_weight = self.model.visual.attnpool.c_proj.weight
            self.c_linear_bias = self.model.visual.attnpool.c_proj.bias
        


    def encode_image(self, x):
        if self.if_vit :
            return self.encode_image_vit(x)
        else :
            return self.encode_image_rn(x)
        
    def get_clsToken(self,x):
        if self.if_vit :    
            x = x[:, 0, :]
        else :
            x ,_= self.model.visual.attnpool(x, if_pos=False)
        return x

    def encode_image_vit(self, x):
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        #if self.model.visual.proj is not None:
           # x = x @ self.model.visual.proj

        return self.model.visual.ln_post(x) #[B,1+L,D]
    
    def encode_image_rn(self, x):
        def stem(x):
            for conv, bn in [(self.visual_encoder.conv1, self.visual_encoder.bn1), \
                (self.visual_encoder.conv2, self.visual_encoder.bn2), (self.visual_encoder.conv3, self.visual_encoder.bn3)]:
                x = self.visual_encoder.relu(bn(conv(x)))
            x = self.visual_encoder.avgpool(x)
            return x

        x = x.type(self.visual_encoder.conv1.weight.dtype)
        x = stem(x)
        x = self.visual_encoder.layer1(x)
        x = self.visual_encoder.layer2(x)
        x = self.visual_encoder.layer3(x)
        x = self.visual_encoder.layer4(x)
        return x
    
    @torch.no_grad()
    def getFeatAndCls(self,image):
        image_features=self.encode_image(image)
        image_feature_ = self.get_clsToken(image_features)
            
        if not self.if_vit:
            b, c, h, w = image_features.shape
            image_features = image_features.reshape(b, c, h * w).permute(2, 0, 1)#LBD
            image_features = F.linear(image_features, self.v_linear_weight, self.v_linear_bias)
            image_features = F.linear(image_features, self.c_linear_weight, self.c_linear_bias)
            image_features=image_features.permute(1, 0, 2) #BLD
        else:
            image_features = image_features[:,1:,:] @ self.model.visual.proj
            image_feature_=image_feature_@ self.model.visual.proj
        return  image_features, image_feature_

    def forward(self, image=None, captions=None, if_test=False,label=None,feat=None,epoch=0):
            if not if_test:
                image_features= feat
                # b, l, d = image_feat.shape
                image_feature_ = image_features[torch.arange(image_features.shape[0]), captions.argmax(dim=-1)]  # BD
 
                #==============add noisy======================
                index_labels = torch.nonzero(label, as_tuple=False)
                mask_head=torch.zeros_like(image_feature_).to(self.device)
                mask_local=torch.zeros_like(image_feature_).to(self.device)
                count = torch.zeros(image_feature_.shape[0],1).to(self.device)
                for index in index_labels:
                    mask_head[index[0]]+=self.filter_head[index[1]].to(self.device)
                    mask_local[index[0]]+=self.filter_local[index[1]].to(self.device)
                    count[index[0],0]+=1  

                mask_head=mask_head/count
                mask_local=mask_local/count
                #print(f'batchsize:{mask_local.shape[0]}')
                #print(f'filter num:{mask_local.sum(dim=-1).mean()}')
                #print(f'filter_local:{self.filter_local.sum(dim=-1).mean()}')
   

                image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)
                image_features= image_features / image_features.norm(dim=-1, keepdim=True)    
                noise=torch.randn(image_feature_.shape)
                image_feature_=image_feature_+(0.03*noise).to(self.device)
                noise=torch.randn(image_features.shape)
                image_features=image_features+(0.03*noise).to(self.device) 
                if epoch>20:
                    image_features=image_features*mask_local[:,None,:]
                    image_feature_=image_feature_*mask_local
                
                


                
            else :
                image_features,image_feature_=self.getFeatAndCls(image)
                image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)
                image_features= image_features / image_features.norm(dim=-1, keepdim=True)            

            
            #================global_logit=================
            logits_head=self.head(3*image_feature_) #B,N

            #================local_logit=================
            text_mask=None
            locla_logit=self.local_head(image_features)
            if captions is not None:
                text_mask = (captions == 0).long() * (-10000)  # BL
                local_attn=F.softmax(locla_logit*50+text_mask[:,:,None],dim=1)
            else:
                local_attn=F.softmax(locla_logit*50,dim=1)
            logits_local=local_attn*locla_logit
            logits_local=torch.sum(logits_local, dim=1)


            return logits_head , logits_local, image_feature_,torch.sum(image_features, dim=1)/image_features.shape[1]


    @torch.no_grad()
    def cal_nosiy_mean(self,clip_model=None):
        clip_model=self.model
        #===============cal img================
        def cal_img():
            img_noisy=torch.rand(32,3,224,224).to(self.device)
            img_noisy = -2.5+5*img_noisy 
            image_feat,image_features=self.getFeatAndCls(img_noisy.type(clip_model.dtype))
            #image_feature_/=image_feature_.norm(dim=-1,keepdim=True)
            return image_feat.mean(dim=1),image_features
        #=============cal text==================

        
        def cal_text():
            prompt_prefix = " ".join(["X"] * 74)
            prompt=prompt_prefix+'.'
            prompt_tokenize=clip.tokenize(prompt).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt_tokenize).type(clip_model.dtype)
                std = 0.01
                tensor_shape = (32, 75, embedding.shape[2])
                noisy = torch.normal(0, std, size=tensor_shape).to(self.device).type(clip_model.dtype)
                embedding = embedding.repeat(32,1,1)
                text_noisy= torch.cat( [embedding[:, :1, :],noisy,embedding[:, 76:, :]],dim=1)
                text_feat=self.text_encoder(text_noisy, prompt_tokenize, if_embedding=True, if_sequence=True)
                text_feature_ = text_feat[torch.arange(text_feat.shape[0]), prompt_tokenize.argmax(dim=-1)]  # BD
                prompt_tokenize[prompt_tokenize != 0] = 1
                text_feat=text_feat*prompt_tokenize[:,:,None]
                text_feat=torch.sum(text_feat,dim=1)/torch.sum(prompt_tokenize,dim=-1)[:,None]
                return text_feat,text_feature_
            
        img_feat = []
        img_feature = []
        text_feat = []
        text_feature = []
        for _ in tqdm(range(100)):
            feat,feature=cal_img() 
            img_feat.append(feat.cpu())
            img_feature.append(feature.cpu())
        for _ in tqdm(range(100)):
            feat,feature=cal_text() 
            text_feat.append(feat.cpu())
            text_feature.append(feature.cpu())

        
        img_feat= torch.stack(img_feat)
        img_feature= torch.stack(img_feature)
        text_feat= torch.stack(text_feat)
        text_feature= torch.stack(text_feature)
        #print(f'{img_feat.shape},{text_feat.shape}')
        img_feat=img_feat.reshape(-1,img_feat.shape[-1])
        img_feature=img_feature.reshape(-1,img_feature.shape[-1])
        text_feat=text_feat.reshape(-1,text_feat.shape[-1])
        text_feature=text_feature.reshape(-1,text_feature.shape[-1])
        diff_feat=img_feat-text_feat
        diff_feature=img_feature-text_feature
        self.std_feat=torch.std(diff_feat, dim=0, keepdim=True)
        self.std_feature=torch.std(diff_feature, dim=0, keepdim=True)
        print(f'feat:{self.std_feat.mean()},\
                feature:{self.std_feature.mean()},')
