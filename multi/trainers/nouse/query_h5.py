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
from trainers.groupBlock import GroupingBlock, QueryBlock,Net
from trainers.model.base import DenseCLIP

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
class query_h5(TrainerX):
   
    def model_inference(self, img,feat):
        return self.model.eval_ori(feat,img)
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
        self.model.QueryBlock.init_token(head_init_weight)

        # noisy
        self.cal_nosiy_mean(cfg,clip_model)
        self.cal_text_mean()
        self.cal_img_mean()
        

        
        print(f'cos_sim:{F.cosine_similarity(self.model.img_feature_mean, self.model.text_feature_mean)}')

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
        data_loader = self.val_loader
        print("calulate image mean on test set")
        if os.path.exists(f'{dataset}_img_feature_mean.pt') and os.path.exists(f'{dataset}_img_feat_mean.pt') :
            self.model.img_feature_mean = torch.load(f'{dataset}_img_feat_mean.pt')
            self.model.img_feat_mean = torch.load(f'{dataset}_img_feat_mean.pt')
            return

        mean_img_feature = 0
        mean_img_feat = 0
        count = 0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image,split_2,split_3, label = self.parse_batch_eval_img(batch)
            img_feat,img_feature=self.model.getFeatAndCls(image)

            #==============norm======================
            image_features= img_feat / img_feat.norm(dim=-1, keepdim=True)
            image_feature_ = img_feature / img_feature.norm(dim=-1, keepdim=True)


            
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
                output, output_local, _, _ = self.model.train_feature(feature,captions)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output_local, _, _ = self.model.train_feature(feature,captions)
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
