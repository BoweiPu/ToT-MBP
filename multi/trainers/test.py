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
from datasets.MyWrappper import TextFeatureDataset
from trainers.model.mask import DenseCLIP, TextEncoder
import h5py
from .utils import *
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




kl_div_loss = nn.KLDivLoss(reduction='batchmean')
@TRAINER_REGISTRY.register()
class test(TrainerX):
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
            output, output_pos, image_features_, text_features_ = self.model_inference(input)
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
        self.model.to(self.device)
        #self.ReBuild_data_loader(cfg,clip_model)
        head_init_weight = self.get_init_head_weight(if_auto=False)
        print(f'head_init_weight:{head_init_weight.shape}')
        self.model.head.weight = Parameter(head_init_weight.clone()).to(self.device)
        self.model.local_head.weight = Parameter(head_init_weight.clone()).to(self.device)
        print(f'filter num:{self.model.filter_local.sum(dim=-1).mean()}')

        print("Turning off gradients in both the image and the text encoder")
        #print(self.model.named_parameters())
        for name, param in self.model.named_parameters():
           
            if   "head" not in name  and  'local_head' not in name:
                param.requires_grad_(False)
       
        
        self.model.to(self.device)
        self.optim = build_optimizer(self.model.head, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("head", self.model.head, self.optim, self.sched)  
        
        
        self.optim = build_optimizer(self.model.local_head, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("local_head", self.model.local_head, self.optim, self.sched)  

        self.scaler = GradScaler() if cfg.TRAINER.Caption.PREC == "amp" else None
        

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        
    def get_init_head_weight(self, if_auto=True):
        self.set_model_mode("eval")
        dataset = self.cfg.DATASET.NAME
        print("calulate text mean on train set")
        data_loader = self.train_loader_x
        mean_text_feature = torch.zeros(len(self.model.classnames),self.model.dim).to(self.device)
        
        count = torch.zeros(len(self.model.classnames),1).to(self.device)
        #count_all = torch.zeros(len(self.model.classnames),1).to(self.device)
        text_feat=[]
        text_label=[]
        #===========获取全部的值========
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            captions,label = self.parse_batch_train(batch)
            with torch.no_grad():
                feat=self.model.text_encoder(captions,captions,if_embedding=False, if_sequence=False)
            #feat, captions,label  = self.parse_batch_text_feature (batch)
            feat=feat/feat.norm(dim=-1, keepdim=True)
            #head = feat[torch.arange(feat.shape[0]), captions.argmax(dim=-1)]
            text_feat.append(feat.detach().clone())
            text_label.append(label)
        #========add============
        text_feat=torch.cat(text_feat,dim=0)
        text_label=torch.cat(text_label,dim=0)
        
        label_counts=text_label.sum(dim=-1)
        #single_label_indices = torch.where(label_counts == 1)[0]
        #single_label_features = text_feat[single_label_indices]
        #single_label_labels = text_label[single_label_indices]
        index_labels_all = torch.nonzero(text_label, as_tuple=False)
        #index_labels = torch.nonzero(single_label_labels, as_tuple=False)
        head_act=mask_new(text_feat,1.5)
        head_local=mask_new(text_feat,1.5)
        
        head_filter= torch.zeros(len(self.model.classnames),self.model.dim).to(self.device)
        local_filter= torch.zeros(len(self.model.classnames),self.model.dim).to(self.device)


        for index in index_labels_all:
            mean_text_feature[index[1]]+=text_feat[index[0]]
            head_filter[index[1]]+=head_act[index[0]]
            local_filter[index[1]]+=head_local[index[0]]
            count[index[1],0]+=1
            

        head_filter=head_filter/count
        local_filter=local_filter/count
        mean_text_feature=mean_text_feature/count    

        head_filter=torch.abs(head_filter)         
        head_filter=torch.exp(head_filter)/torch.exp(head_filter).sum(dim=0,keepdim=True)
        head_filter=head_filter/head_filter.norm(dim=-1, keepdim=True)

        local_filter=torch.abs(local_filter)         
        local_filter=torch.exp(local_filter)/torch.exp(local_filter).sum(dim=0,keepdim=True)        
        local_filter=local_filter/local_filter.norm(dim=-1, keepdim=True)
        
        head_h=torch.quantile(head_filter, self.cfg.MODEL.r)
        head_filter[head_filter>=head_h]=1
        head_filter[head_filter<head_h]=0

        head_l=torch.quantile(local_filter, self.cfg.MODEL.r)
 
        local_filter[local_filter>=head_l]=1
        local_filter[local_filter<head_l]=0

        self.model.filter_head=head_filter
        self.model.filter_local=local_filter
        self.model.text_feature_mean=mean_text_feature
        return mean_text_feature.float()
        
    
    def ReBuild_data_loader(self,cfg,clip_model):
        """
        提前计算全部特征值
        """
        backbone=cfg.MODEL.BACKBONE.NAME
        backbone=backbone.replace("/", "_")
        file_path_text=f'features/{cfg.DATASET.NAME}_{backbone}_text_train.h5'
        encoder=TextEncoder(clip_model).to(self.device)
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
                return encoder(caption,caption,if_embedding=False, if_sequence=True)
            
            for batch_idx, batch in enumerate(tqdm(train_loader_x)):
                caption,label = self.parse_batch_train(batch)
                text_feature=encoder_text(caption )
                save_to_hdf5( count,text_feature.cpu(), caption.cpu(),label.cpu())
                count+=len(text_feature)
               

        train_dataset=TextFeatureDataset(file_path_text)

      
        #===========================rebuild loader=========================
        train_loader_x=torch.utils.data.DataLoader(train_dataset,
                            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            drop_last=False ,
                            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA))
        self.train_loader_x=train_loader_x

     
    def forward_backward(self, batch):
        #self.test('test')
        #feat, captions,label  = self.parse_batch_text_feature (batch)
        captions,label = self.parse_batch_train(batch)
        with torch.no_grad():
            feat=self.model.text_encoder(captions,captions,if_embedding=False, if_sequence=True)

        output, output_local, image_feature_, image_feat = self.model(None, captions=captions,feat=feat,label=label,epoch=self.epoch)
        loss = ASL_loss(output, label) + ASL_loss(output_local, label.float())

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),  
        }

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