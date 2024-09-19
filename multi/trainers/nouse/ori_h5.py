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
from datasets.MyWrappper import ImgFeatDataset,  Split_DatasetWrapper, TextFeatureDataset

from .utils import soft_cross_entropy, softmax_sigmoid_BCEloss, \
    norm_logits_BCEloss, sigmoid_focal_loss, sigmoid_ASL_loss, ranking_loss, ASL_loss
_tokenizer = _Tokenizer()
import pdb
from torch.nn.parameter import Parameter
from .imagenet_templates import *

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


class PromptLearner(nn.Module):
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
            
            if cfg.TRAINER.Caption.CSC:
                print("Initializing class-specific double contexts")
                ctx_vectors_double = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_double = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_double, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial double context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_double = nn.Parameter(ctx_vectors_double)  # to be optimized
        
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
        ctx_double = self.ctx_double
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        if ctx_double.dim() == 2:
            ctx_double = ctx_double.unsqueeze(0).expand(self.n_cls, -1, -1)

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
            if neg_prompt_wcls:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,     # (n_cls, n_ctx, dim)
                        suffix_nocls,  # (n_cls, *, dim)
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

        return prompts, prompts_neg, self.temperature, self.spatial_T, self.ranking_scale


class DenseCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, return_interm_layers=False):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.classnames = classnames
        self.text_encoder = TextEncoder(clip_model)

        self.model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.head = nn.Linear(512, 20, bias=True)
        self.local_head= nn.Linear(512, 20, bias=True)
        self.drop_rate = 0.0
        self.add_std = 0.1
        self.mul_left = 0.8
        self.mul_right = 1.2
        self.drop = nn.Dropout(self.drop_rate)
        
        #self.groupBlock.half()
        backbone_name = cfg.MODEL.BACKBONE.NAME
        self.if_vit=not backbone_name.startswith('R')

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
            x =self.model.visual.ln_post(x[:, 0, :])
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

        return x #[B,1+L,D]
    
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
    
    def forward(self, image_feat, captions):
            image_feature_ = image_feat[torch.arange(image_feat.shape[0]), captions.argmax(dim=-1)]  # BD
            image_features = image_feat.permute(1, 0, 2)  # LBD
            # ===============================================================

            prompts, prompts_double, temperature, spatial_T, rk_scale = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features_neg = self.text_encoder(prompts_double, tokenized_prompts)

            image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)
            
            # mask irrelavent tokens
            text_mask = (captions == 0).long() * (-10000)  # BL

            logit_scale = temperature.exp()  # rk_scale
            logit_scale = logit_scale if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50 # temperature.exp()  # self.logit_scale.exp()
            logits_ = logit_scale * image_feature_ @ text_features.t()   # B * C,  cls * C, = B * cls
            logits_neg = image_features @ text_features_neg.t()    #  L * B * C,  cls * C,  L * B * cls
            logits_neg = logits_neg.permute(2, 1, 0) + text_mask[None, :, :]
            logits_neg = logits_neg.permute(2, 1, 0) # L * B * cls

            tmp_scale = spatial_T.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_text
            prob_spatial = torch.nn.functional.softmax(logits_neg * tmp_scale, dim=0)
            logits_local = torch.sum(logit_scale * logits_neg * prob_spatial, dim=0)
            
            
            return logits_, logits_local, image_feature_,torch.sum(image_features.permute(1, 0, 2),dim=1)/image_features.shape[0]


    def eval(self,image_features,image_feature_):
            prompts, prompts_double, temperature, spatial_T, rk_scale = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features_neg = self.text_encoder(prompts_double, tokenized_prompts)

            image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)

            logit_scale = temperature.exp()  # rk_scale
            logit_scale = logit_scale if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50
            logits_ = logit_scale * image_feature_ @ text_features.t()   # B * C,  cls * C, = B * cls
            logits_neg = image_features @ text_features_neg.t()    #  HW * B * C,  cls * C,  HW * B * cls
           
            tmp_scale = spatial_T.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_image  # 5 #
            prob_spatial = torch.nn.functional.softmax(logits_neg * tmp_scale, dim=1)
            logits_local = torch.sum(logit_scale * logits_neg * prob_spatial, dim=1)

            #print(f'logits_:{logits_.shape},logits_local:{logits_local.shape},image_features:{image_features.shape}')
            return logits_, logits_local, logits_neg, image_features @ text_features.t()  # compare additional branch with global proxy

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


@TRAINER_REGISTRY.register()
class ori_h5(TrainerX):
   
    def model_inference(self, img,feat):
        return self.model.eval(feat,img)
        # return self.model(None, input)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")
            
     
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            img, img_2,img_3,label,feat= self.parse_batch_img_feature(batch)
            # output = self.model_inference(input)
            output, output_pos, image_features_, text_features_ = self.model_inference(img , feat)
            self.evaluator.process(output, label, output_pos)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test: # and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")  # self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")

    def check_cfg(self, cfg):
        assert cfg.TRAINER.Caption.PREC in ["fp16", "fp32", "amp"]

    def ReBuild_data_loader(self,cfg):
        """
        提前计算全部特征值
        """
        backbone=cfg.MODEL.BACKBONE.NAME
        backbone=backbone.replace("/", "_")
        file_path_text=f'features/{cfg.DATASET.NAME}_{backbone}_text_train.h5'
        print(file_path_text)
        if not os.path.exists(file_path_text):
            train_loader_x = self.train_loader_x
            count=0
            print(f'caculate text feature to :{file_path_text}')
            AllBatch=len(self.dm.dataset.train_x)
            with h5py.File(file_path_text,'w') as f:            
                f.create_dataset('feature', (AllBatch,77, self.model.dim), dtype='f')
                f.create_dataset('caption', (AllBatch, 77 ), dtype='f')
                f.create_dataset('label', (AllBatch, self.dm.num_classes), dtype='f')

            def save_to_hdf5( index,feature, caption,label):
                with h5py.File(file_path_text, 'r+') as hf:
                    hf['feature'][index:index+len(feature)] = feature.numpy()
                    hf['caption'][index:index+len(caption)] = caption.numpy()
                    hf['label'][index:index+len(label)] = label.numpy()
            @torch.no_grad()
            def encoder_text(caption ):
                return self.model.text_encoder(caption,caption,if_embedding=False, if_sequence=True)
            
            for batch_idx, batch in enumerate(tqdm(train_loader_x)):
                caption,label = self.parse_batch_train_text(batch)
                text_feature=encoder_text(caption )
                save_to_hdf5( count,text_feature.cpu(), caption.cpu(),label.cpu())
                count+=len(text_feature)
                

        #======================img===========================
        file_path_img=f'features/{cfg.DATASET.NAME}_{backbone}_img_test.h5'
        
        if not os.path.exists(file_path_img):
            print(f'caculate img feature to:{file_path_img}')
            count=0
            if self.val_loader is not None:
                test_loader = self.val_loader
                AllBatch=len(self.dm.dataset.val)
            else:
                test_loader = self.test_loader
                AllBatch=len(self.dm.dataset.test)
             
            with h5py.File(file_path_img,'w') as f:            
                #dset = f.create_dataset('features', (AllBatch, self.model.dim), dtype='f') 
                f.create_dataset('img', (AllBatch, self.model.dim), dtype='f')
                f.create_dataset('img_2', (AllBatch, 4 , self.model.dim ), dtype='f')
                f.create_dataset('img_3', (AllBatch, 9 , self.model.dim), dtype='f') 
                f.create_dataset('label', (AllBatch,  self.dm.num_classes), dtype='f')
                f.create_dataset('feat', (AllBatch, self.model.img_feat_shape[1], self.model.dim), dtype='f')

            def save_to_hdf5( index,img, img_2, img_3, feat,label):
                with h5py.File(file_path_img, 'r+') as hf:
                    hf['img'][index:index+len(img)] = img.numpy()
                    hf['img_2'][index:index+len(img_2)] = img_2.numpy()
                    hf['img_3'][index:index+len(img_3)] = img_3.numpy()
                    hf['label'][index:index+len(label)] = label.numpy()
                    hf['feat'][index:index+len(label)] = feat.numpy()

            for batch_idx, batch in enumerate(tqdm(test_loader)):
                image,split_2,split_3, label = self.parse_batch_eval_img(batch)
                img_feat,img_feature=self.model.getFeatAndCls(image)
                img_feature_2_=[]
                for split in torch.unbind(split_2, dim=1):
                    _,feature=self.model.getFeatAndCls(split)
                    img_feature_2_.append(feature)
                img_feature_2=torch.stack(img_feature_2_,dim=1)

                img_feature_3_=[]
                for split in torch.unbind(split_3, dim=1):
                    _,feature=self.model.getFeatAndCls(split)
                    img_feature_3_.append(feature)

                img_feature_3=torch.stack(img_feature_3_,dim=1)
                
                save_to_hdf5(count,img_feature.cpu(),img_feature_2.cpu(),img_feature_3.cpu(),img_feat.cpu(),label.cpu())
                count+=len(img_feature)
        
        train_dataset=TextFeatureDataset( file_path_text)
        test_dataset=ImgFeatDataset( file_path_img)
      
        #===========================rebuild loader=========================
        train_loader_x=torch.utils.data.DataLoader(train_dataset,
                            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            drop_last=True ,
                            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA))
        test_loader=torch.utils.data.DataLoader(test_dataset,
                            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            drop_last=True ,
                            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA))
        
        self.train_loader_x=train_loader_x
        self.val_loader=test_loader
        self.test_loader=test_loader
     
    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (except self.dm).
        """
        dm = DataManager(self.cfg,dataset_wrapper=Split_DatasetWrapper)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        print('==================== Building model in Caption_distill_double ======================')
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.Caption.PREC == "fp32" or cfg.TRAINER.Caption.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        # self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model = DenseCLIP(cfg, classnames, clip_model)
        
        print("Turning off gradients in both the image and the text encoder")
        #print(self.model.named_parameters())
        for name, param in self.model.named_parameters():
           
            if "prompt_learner" not in name \
                and "head" not in name \
                and "local_head" not in name \
                and "QueryBlock" not in name :
                param.requires_grad_(False)
            
       
        
        
        self.model.to(self.device)
        # head init
        head_init_weight = self.get_init_head_weight(if_auto=False)
        print(f'head_init_weight:{head_init_weight.shape}')
        self.model.head.weight = Parameter(head_init_weight).to(self.device)
        self.model.local_head.weight = Parameter(head_init_weight).to(self.device)

        # NOTE: only give prompt_learner to the optimizer     
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        
        self.scaler = GradScaler() if cfg.TRAINER.Caption.PREC == "amp" else None
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        #====================rebuild data loader==============
        self.ReBuild_data_loader(cfg)
 
    @torch.no_grad()
    def cal_nosiy_mean(self,cfg,clip_model):
        
        #===============cal img================
        img_noisy=torch.randn(1,3,224,224).to(self.device)
        img_noisy = (img_noisy * torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1).to(self.device)) + torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1).to(self.device)
        image_features=self.model.encode_image(img_noisy)
        image_feature_ = self.model.get_clsToken(image_features)

        if not self.model.if_vit:
            b, c, h, w = image_features.shape
            image_features = image_features.reshape(b, c, h * w).permute(2, 0, 1)#LBD
            image_features = F.linear(image_features, self.model.v_linear_weight, self.model.v_linear_bias)
            image_features = F.linear(image_features, self.model.c_linear_weight, self.model.c_linear_bias)
            image_features=image_features.permute(1, 0, 2) #BLD
        else:
            image_features = image_features[:,1:,:] @ self.model.model.visual.proj
            image_feature_=image_feature_ @ self.model.model.visual.proj

        image_features= image_features / image_features.norm(dim=-1, keepdim=True)

        self.model.img_feat_shape =image_features.shape

        image_features=torch.sum(image_features,dim=1)/image_features.shape[1]

        image_feature_=image_feature_ / image_feature_.norm(dim=-1, keepdim=True)

        self.model.img_feature_mean = image_feature_.float() 
        self.model.img_feat_mean = image_features.float() 
        
        #=============cal text==================

        std = 0.02
        tensor_shape = (1, 75, 512)
        noisy = torch.normal(0, std, size=tensor_shape).to(self.device).type(clip_model.dtype)
        prompt_prefix = " ".join(["X"] * 74)
        prompt=prompt_prefix+'.'
        prompt_tokenize=clip.tokenize(prompt).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt_tokenize).type(clip_model.dtype)
        text_noisy= torch.cat( [embedding[:, :1, :],noisy,embedding[:, 76:, :]],dim=1)

        text_feat = self.model.text_encoder(text_noisy, None, if_embedding=True, if_sequence=True) 
        text_feature_ = text_feat[torch.arange(text_feat.shape[0]), prompt_tokenize.argmax(dim=-1)]  # BD
        text_feature_ = text_feature_ / text_feature_.norm(dim=-1, keepdim=True)
        prompt_tokenize[prompt_tokenize != 0] = 1
        text_feat=text_feat*prompt_tokenize[:,:,None]
        text_feat=torch.sum(text_feat,dim=1)/torch.sum(prompt_tokenize,dim=-1)[:,None]

        self.model.text_feature_mean = text_feature_.float()
        self.model.text_feat_mean =text_feat.float() 

        print(f'image_features,:{image_features.shape},mean_text_feat:{text_feat.shape},cos_sim:{F.cosine_similarity(image_feature_, text_feature_)}')

    @torch.no_grad()
    def cal_img_mean(self):
        self.set_model_mode("eval")
        dataset = self.cfg.DATASET.NAME
        data_loader = self.test_loader
        print("calulate image mean on test set")
        if os.path.exists(f'{dataset}_img_feature_mean.pt') and os.path.exists(f'{dataset}_img_feat_mean.pt') :
            self.model.img_feature_mean = torch.load(f'{dataset}_img_feat_mean.pt')
            self.model.img_feat_mean = torch.load(f'{dataset}_img_feat_mean.pt')
            return

        mean_img_feature = 0
        mean_img_feat = 0
        count = 0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image,split_2,split_3, label = self.parse_batch_train_text(batch)
            image_features=self.model.encode_image(image)
            image_feature_ = self.model.get_clsToken(image_features)
            
            if not self.model.if_vit:
                b, c, h, w = image_features.shape
                image_features = image_features.reshape(b, c, h * w).permute(2, 0, 1)#LBD
                image_features = F.linear(image_features, self.model.v_linear_weight, self.model.v_linear_bias)
                image_features = F.linear(image_features, self.model.c_linear_weight, self.model.c_linear_bias)
                image_features=image_features.permute(1, 0, 2) #BLD
            else:
                image_features = image_features[:,1:,:] @ self.model.model.visual.proj
                image_feature_ = image_feature_@ self.model.model.visual.proj

            #==============norm======================
            image_features= image_features / image_features.norm(dim=-1, keepdim=True)
            image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)


            
            mean_img_feature += image_feature_.mean(dim=0, keepdim=True)

            image_features_sum=torch.sum(image_features,dim=1)/image_features.shape[1]
            mean_img_feat+= image_features_sum.mean(dim=0, keepdim=True)
            
            count+=1
        self.model.img_feat_shape =image_features.shape
        self.model.img_feature_mean = mean_img_feature.float() / count
        self.model.img_feat_mean = mean_img_feat.float() / count
        

        torch.save(self.model.img_feature_mean, f'{dataset}_img_feature_mean.pt')
        torch.save(self.model.img_feat_mean, f'{dataset}_img_feat_mean.pt')

    @torch.no_grad()
    def cal_text_mean(self):
        self.set_model_mode("eval")
        dataset = self.cfg.DATASET.NAME
        print("calulate text mean on train set")
        if os.path.exists(f'{dataset}_text_feature_mean.pt') and os.path.exists(f'{dataset}_text_feat_mean.pt') :
            self.model.text_feature_mean = torch.load(f'{dataset}_text_feat_mean.pt')
            self.model.text_feat_mean = torch.load(f'{dataset}_text_feat_mean.pt')
            return
        
        
        
        data_loader = self.train_loader_x
        mean_text_feat=0
        mean_text_feature = 0
        count = 0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            captions, label = self.parse_batch_train_text(batch)
            image_feat = self.model.text_encoder(captions, None, if_embedding=False, if_sequence=True) 
            image_feature_ = image_feat[torch.arange(image_feat.shape[0]), captions.argmax(dim=-1)]  # BD
            image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)
            #=========feat===========
            captions[captions != 0] = 1
            image_feat=image_feat*captions[:,:,None]

            text_feat=torch.sum(image_feat,dim=1)/torch.sum(captions,dim=-1)[:,None]
            #========mean============
            mean_text_feature += image_feature_.mean(dim=0, keepdim=True)
            mean_text_feat+=text_feat.mean(dim=0, keepdim=True)

            count+=1
        
        self.model.text_feature_mean = mean_text_feature.float() / count
        self.model.text_feat_mean =mean_text_feat.float() / count
        
        
        torch.save(self.model.text_feature_mean, f'{dataset}_text_feature_mean.pt')
        torch.save(self.model.text_feat_mean, f'{dataset}_text_feat_mean.pt')
 
    def get_init_head_weight(self, if_auto=True):
        if if_auto:
            with torch.no_grad():
                prompts, temperature, spatial_T, rk_scale = self.model.prompt_learner()
                tokenized_prompts = self.model.tokenized_prompts
                text_features = self.model.text_encoder(prompts, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features
        
        else:
          
            templates = IMAGENET_TEMPLATES_SELECT_3

            classnames = self.model.classnames
            print(classnames)
           
            dataset_name = 'COCO'
            try:
                templates.append(CUSTOM_TEMPLATES[dataset_name])
                
            except:
                print('!! WARNING: Not found template for {}'.format(dataset_name))
                templates = "a photo of a {}."

            num_temp = len(templates)
            print(f"Prompt ensembling (n={num_temp})")
            #print(templates)
            
            mean_text_features = 0
            with torch.no_grad():
                for i, temp in enumerate(templates):
                    prompts = [temp.format(c.replace("_", " ")) for c in classnames]
                    prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
                    text_features = self.model.model.encode_text(prompts)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    mean_text_features = mean_text_features + text_features
                mean_text_features = mean_text_features / num_temp
                mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
            print("yes!")
            return mean_text_features
        
    def forward_backward(self, batch):
      
        feature,captions, label = self.parse_batch_text_feature(batch)
    
        prec = self.cfg.TRAINER.Caption.PREC
        if prec == "amp":
            with autocast():
                output, output_local, _, _ = self.model(feature,captions)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output_local, _, _ = self.model(feature,captions)
            if   self.cfg.TRAIN.LOSSFUNC == 'sigmoid':
                loss = norm_logits_BCEloss(output, label.float()) + norm_logits_BCEloss(output_local, label.float())
            elif self.cfg.TRAIN.LOSSFUNC == 'focal':
                loss = sigmoid_focal_loss(output, label)
            elif self.cfg.TRAIN.LOSSFUNC == 'asl':
                loss = ASL_loss(output, label) + ASL_loss(output_local, label.float())
            elif self.cfg.TRAIN.LOSSFUNC == 'ranking':
                loss = ranking_loss(output, label)
            elif self.cfg.TRAIN.LOSSFUNC == 'double_ranking':
                loss = ranking_loss(output, label, scale_ = 1.0, margin_ = 1)\
                    + ranking_loss(output_local, label, scale_ = 1.0, margin_ = 1)
               
            else:
                loss = soft_cross_entropy(output, label)
                

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),            
        }
       
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_eval_img(self, batch):
        input = batch["img"]
        split_3=batch['splits_3']
        split_2=batch['splits_2']
        label = batch["label"]
        img = input.to(self.device)
        label = label.to(self.device)
        split_3=split_3.to(self.device)
        split_2=split_2.to(self.device)
        return img,split_2,split_3, label
    
    def parse_batch_train_text(self, batch):
        
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def parse_batch_text_feature(self, batch):
        feature = batch['feature']

        caption = batch["caption"]
        #print(caption)
        label = batch['label']
        #print(label)
        feature = feature.to(self.device)
        caption = caption.to(self.device)
        label = label.to(self.device)
        return feature, caption,label
    
    def parse_batch_img_feature(self, batch):
        img = batch["img"]
        img_2 = batch["img2"]
        img_3 = batch["img3"]
        feat = batch["feat"]
        label = batch["label"]
        img = img.to(self.device)
        img_2 = img_2.to(self.device)
        img_3 = img_3.to(self.device)
        label = label.to(self.device)
        feat = feat.to(self.device) 
        return img, img_2,img_3,label,feat

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
            print(state_dict)
