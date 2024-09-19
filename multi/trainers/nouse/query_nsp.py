import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.models._utils import IntermediateLayerGetter

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
from trainers.groupBlock import GroupingBlock, QueryBlock
from trainers.model.nsp import DenseCLIP

from .utils import soft_cross_entropy, softmax_sigmoid_BCEloss, \
    norm_logits_BCEloss, sigmoid_focal_loss, sigmoid_ASL_loss, ranking_loss, ASL_loss
_tokenizer = _Tokenizer()
import pdb
from torch.nn.parameter import Parameter
from .imagenet_templates import *


def min_max_scaling_torch(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    return (data - min_val) / (max_val - min_val)

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



@TRAINER_REGISTRY.register()
class query_nsp(TrainerX):
    def model_inference(self, input, mean=None):
        return self.model(input, if_test=True)
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
            input, label = self.parse_batch_test(batch)
            # output = self.model_inference(input)
            output, output_pos, _, _ = self.model_inference(input)
            self.evaluator.process(output, label, output_pos)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def check_cfg(self, cfg):
        assert cfg.TRAINER.Caption.PREC in ["fp16", "fp32", "amp"]

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
        self.model.QueryBlock.init_token(head_init_weight)
        
        
        # NOTE: only give prompt_learner to the optimizer     
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        self.optim = build_optimizer(self.model.head, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("head", self.model.head, self.optim, self.sched)  
        
        self.optim = build_optimizer(self.model.local_head, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("local_head", self.model.local_head, self.optim, self.sched)  

        self.optim = build_optimizer(self.model.QueryBlock, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("QueryBlock", self.model.QueryBlock, self.optim, self.sched)  

        self.scaler = GradScaler() if cfg.TRAINER.Caption.PREC == "amp" else None
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.cal_nosiy_mean(cfg,clip_model)
        self.cal_img_mean()
        self.cal_text_mean()

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
        
        print("calulate image mean on test set")
        if os.path.exists(f'mean/sp/{dataset}_img_feature_mean_sp.pt') and os.path.exists(f'mean/sp/{dataset}_img_feat_mean_sp.pt') :
            self.model.img_feature_mean = torch.load(f'mean/sp/{dataset}_img_feat_mean_sp.pt')
            self.model.img_feat_mean = torch.load(f'mean/sp/{dataset}_img_feat_mean_sp.pt')
            return
        
        data_loader = self.val_loader
        mean_img_feat=torch.zeros(len(self.model.classnames),self.model.dim).to(self.device)
        mean_img_feature = torch.zeros(len(self.model.classnames),self.model.dim).to(self.device)
        count = torch.zeros(len(self.model.classnames),1).to(self.device)

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            img, label= self.parse_batch_train(batch)
            img_feat,img_feature=self.model.getFeatAndCls(img)
            #==============norm======================
            image_features= img_feat / img_feat.norm(dim=-1, keepdim=True)
            image_feature_ = img_feature / img_feature.norm(dim=-1, keepdim=True)

            image_features_sum=torch.sum(image_features,dim=1)/image_features.shape[1]

            #========add============
            index_labels = torch.nonzero(label, as_tuple=False)
            for index in index_labels:
                mean_img_feature[index[1]]+=image_feature_[index[0]]
                mean_img_feat[index[1]] +=image_features_sum[index[0]]
                count[index[1],0]+=1

        self.model.img_feat_shape =image_features.shape
        self.model.img_feature_mean = mean_img_feature.float() / count
        self.model.img_feat_mean = mean_img_feat.float() / count
        

        torch.save(self.model.img_feature_mean, f'mean/sp/{dataset}_img_feature_mean_sp.pt')
        torch.save(self.model.img_feat_mean, f'mean/sp/{dataset}_img_feat_mean_sp.pt')

    @torch.no_grad()
    def cal_text_mean(self):
        self.set_model_mode("eval")
        dataset = self.cfg.DATASET.NAME
        print("calulate text mean on train set")
        
        if os.path.exists(f'mean/sp/{dataset}_text_feature_mean_sp.pt') and os.path.exists(f'mean/sp/{dataset}_text_feat_mean_sp.pt') :
            self.model.text_feature_mean = torch.load(f'mean/sp/{dataset}_text_feat_mean_sp.pt')
            self.model.text_feat_mean = torch.load(f'mean/sp/{dataset}_text_feat_mean_sp.pt')
            return
        
        data_loader = self.train_loader_x
        mean_text_feat=torch.zeros(len(self.model.classnames),self.model.dim).to(self.device)
        mean_text_feature = torch.zeros(len(self.model.classnames),self.model.dim).to(self.device)
        count = torch.zeros(len(self.model.classnames),1).to(self.device)
        @torch.no_grad()
        def encoder_text(caption ):
                return self.model.text_encoder(caption,caption,if_embedding=False, if_sequence=True)

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            captions,label  = self.parse_batch_train (batch)
            image_feat=encoder_text(captions )
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            image_feature_ = image_feat[torch.arange(image_feat.shape[0]), captions.argmax(dim=-1)]  # BD
            #=========feat===========
            captions[captions != 0] = 1
            image_feat=image_feat*captions[:,:,None]
            text_feat=torch.sum(image_feat,dim=1)/torch.sum(captions,dim=-1)[:,None]
            #========add============
            index_labels = torch.nonzero(label, as_tuple=False)
            for index in index_labels:
                mean_text_feature[index[1]]+=image_feature_[index[0]]
                mean_text_feat[index[1]] +=text_feat[index[0]]
                count[index[1],0]+=1
        
        self.model.text_feature_mean = mean_text_feature.float() / count
        self.model.text_feat_mean =mean_text_feat.float() / count
        
        
        torch.save(self.model.text_feature_mean, f'mean/sp/{dataset}_text_feature_mean_sp.pt')
        torch.save(self.model.text_feat_mean, f'mean/sp/{dataset}_text_feat_mean_sp.pt')
 
 
    def get_init_head_weight(self, if_auto=True):
        if if_auto:
            with torch.no_grad():
                prompts, temperature, spatial_T, rk_scale = self.model.prompt_learner()
                tokenized_prompts = self.model.tokenized_prompts
                text_features = self.model.text_encoder(prompts, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features
        
        else:
            # temp
            templates = IMAGENET_TEMPLATES_SELECT_3
#             templates = []
    
#             self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            classnames = self.model.classnames
            print(classnames)
            
            #dataset_name = self.dataset_name
            dataset_name = 'COCO'
            try:
#                 templates = CUSTOM_TEMPLATES[dataset_name]
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
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.Caption.PREC
        if prec == "amp":
            with autocast():
                output, output_local, _, _ = self.model(image,label=label)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output_local, image_feature_, image_feat = self.model(None, image,label=label)
            if   self.cfg.TRAIN.LOSSFUNC == 'sigmoid':
                loss = norm_logits_BCEloss(output, label.float()) + norm_logits_BCEloss(output_local, label.float())
            elif self.cfg.TRAIN.LOSSFUNC == 'focal':
                loss = sigmoid_focal_loss(output, label)
            elif self.cfg.TRAIN.LOSSFUNC == 'asl':
                loss = ASL_loss(output, label) + ASL_loss(output_local, label.float())
            elif self.cfg.TRAIN.LOSSFUNC == 'ranking':
                loss = ranking_loss(output, label)
            elif self.cfg.TRAIN.LOSSFUNC == 'double_ranking':
                loss = ranking_loss(output, label, scale_ = 1.0, margin_ = 1) + ranking_loss(output_local, label, scale_ = 1.0, margin_ = 1)
            else:
                loss = soft_cross_entropy(output, label)
                #loss = soft_cross_entropy(output, label)

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "cos_G&L":torch.sum(F.cosine_similarity(output, output_local))/output.shape[0]
            #"cos_G&L":torch.sum(F.cosine_similarity(image_feature_, image_feat))/output.shape[0]
            
        }
        needPrint=f'logit:{output},\nlogit_local:{output_local}'
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
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
            print(state_dict)
