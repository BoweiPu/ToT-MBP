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
from trainers.model.groupBlock import Net, QueryBlock

from ..utils import soft_cross_entropy, softmax_sigmoid_BCEloss, \
    norm_logits_BCEloss, sigmoid_focal_loss, sigmoid_ASL_loss, ranking_loss, ASL_loss
_tokenizer = _Tokenizer()
import pdb
from torch.nn.parameter import Parameter
from ..imagenet_templates import *

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' was created.")
    else:
        print(f"Directory '{directory}' already exists.")

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


class PromptLearnerSingle(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.Caption.N_CTX
        ctx_init = cfg.TRAINER.Caption.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.Caption.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        
        temperature = torch.tensor(3.0, dtype=dtype)  #  exp(3.91) = 50
        self.temperature = nn.Parameter(temperature)
        spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)
        ranking_scale = torch.tensor(4.0, dtype=dtype)  # 20
        self.ranking_scale = nn.Parameter(ranking_scale)

        # sigmoid_shift = torch.tensor(0.25, dtype=dtype)
        # self.sigmoid_shift = nn.Parameter(sigmoid_shift)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        
        # class agnostic token suffix
        prompts_nocls = [prompt_prefix + "."] * len(classnames)
        tokenized_prompts_nocls = torch.cat([clip.tokenize(p) for p in prompts_nocls])
        with torch.no_grad():
            embedding_nocls = clip_model.token_embedding(tokenized_prompts_nocls).type(dtype)
        self.register_buffer("token_suffix_nocls", embedding_nocls[:, 1 + n_ctx :, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.Caption.CLASS_TOKEN_POSITION

        
    def forward(self, neg_prompt_wcls=True):
        """
        Returns current learned ctx embeddings, concated with cls word embeddings.
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        suffix_nocls = self.token_suffix_nocls

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, self.temperature, self.spatial_T, self.ranking_scale


class DenseCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, net=None,return_interm_layers=False):
        super().__init__()
        self.prompt_learner = PromptLearnerSingle(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
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
        self.drop = nn.Dropout(self.drop_rate)
        
        #self.groupBlock.half()
        backbone_name = cfg.MODEL.BACKBONE.NAME
        self.if_vit=not backbone_name.startswith('R')
        self.filter=nn.Parameter(torch.eye(len(classnames),len(classnames)))
        self.dim=self.model.text_projection.shape[1]
        self.head = nn.Linear(self.dim, len(classnames), bias=True)
        self.local_head= nn.Linear(self.dim,len(classnames), bias=True)
        self.QueryBlock=QueryBlock(dtype=self.dtype,num_token=16,dim=self.dim,low_dim=len(classnames))
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

        if net is not None:
            self.net=net
        else:
            self.net=Net(dim=self.dim)

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

    def forward(self, image=None, captions=None, if_test=False, label=None):
        if if_test:
            image_features,image_feature_=self.getFeatAndCls(image)
            #==============norm======================
            image_features= image_features / image_features.norm(dim=-1, keepdim=True)
            image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)

            logits_head=self.head(self.QueryBlock.scale2 *image_feature_) #B,N

            local_attn=self.QueryBlock(x=image_features,mask=None) #B,L,N

            logits_local=self.local_head(image_features)*local_attn

            logits_local=torch.sum(logits_local , dim=1)
            
        else:

            feature= self.text_encoder(captions, None, if_embedding=False, if_sequence=True)
            # b, l, d = image_feat.shape
            text_feature_ = feature[torch.arange(feature.shape[0]), captions.argmax(dim=-1)]  # BD

           #==============norm======================
            text_feature_ = text_feature_ / text_feature_.norm(dim=-1, keepdim=True)
            feature= feature / feature.norm(dim=-1, keepdim=True)
            
            #==============add noisy======================
            index_labels = torch.nonzero(label, as_tuple=False)
            shift_feature_=torch.zeros(text_feature_.shape).to(self.device)
            shift_features=torch.zeros(feature.shape).to(self.device)
            for index in  index_labels:
                shift_feature_[index[0]]+=- self.text_feature_mean[index[1]]
                shift_features[index[0]]+=- self.text_feat_mean[index[1]]

            noise=(self.cfg.MODEL.noisy_rate* torch.randn(text_feature_.shape)+1).to(self.device)
            text_feature_ =shift_feature_.to(self.device)*noise+text_feature_

            noise= (self.cfg.MODEL.noisy_rate*torch.randn(feature.shape)+1).to(self.device)
            feature =shift_features.to(self.device)*noise+feature

            #================global_logit=================
            logits_head=self.head(self.QueryBlock.scale2 *text_feature_) #B,N

            #================local_logit=================
            text_mask = (captions == 0).long() * (-10000)  # BL
        
            local_attn=self.QueryBlock(x=feature,mask=text_mask) #B,L,N

            logits_local=self.local_head(feature)*local_attn

            logits_local=torch.sum(logits_local, dim=1)


        return logits_head,logits_local,None,None
