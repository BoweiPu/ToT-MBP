import os
from os.path import join
from re import L
import pickle5 as pickle
import random
from scipy.io import loadmat
from collections import defaultdict
import torch
import json
from tqdm import tqdm
from clip import clip
from clip.model import convert_weights
from trainers.coop import load_clip_to_cpu

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing
from pycocotools.coco import COCO

from .data_helpers import *

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

object_categories = coco_object_categories
classname_synonyms = coco_classname_synonyms

clsname2idx_ = {}
nameset_compound = set()
nameset = set()
for idx, synset in enumerate(classname_synonyms):
    for n in synset:
        clsname2idx_[n] = idx

        if ' ' in n:
            nameset_compound.add(n)
            m = n.replace(' ', '')
            clsname2idx_[m] = idx
            nameset.add(m)
        else:
            nameset.add(n)

@DATASET_REGISTRY.register()
class COCO2014_gen(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = 'COCO'
        cls_num = len(object_categories)
    
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        coco_instance_json_file = os.path.join(self.dataset_dir, "annotations/instances_val2014.json")

        coco = COCO(coco_instance_json_file)
        self.valset_ids = coco.getImgIds()
        

        instance_info = {}
        with open(coco_instance_json_file, 'r') as f:
            instance_info = json.load(f)

        clsid2clsidx = {}
        clsidx2clsid = {}
        clsid2clsname = {}
        for idx, cat_info in enumerate(instance_info["categories"]):
            clsid2clsidx[cat_info['id']] = idx
            clsidx2clsid[idx] = cat_info['id']
            clsid2clsname[cat_info['id']] = cat_info['name']

        test_imgdir = [self.dataset_dir + '/val2014/{}'.format(coco.loadImgs(ids = imgid)[0]['file_name']) for imgid in self.valset_ids]
        test_label = torch.zeros((len(self.valset_ids), cls_num), dtype=torch.long)
        for idx, imgid in enumerate(self.valset_ids):
            annIds = coco.getAnnIds(imgIds = imgid)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                tmp_idx = clsid2clsidx[ann['category_id']]
                test_label[idx, tmp_idx] = 1

        test = []
        for i in range(len(self.valset_ids)):
            item_ = Datum(impath=test_imgdir[i], label=test_label[i], classname='')
            test.append(item_)


        json_name_all=f"captions/coco2014.json"
        with open(json_name_all, 'r') as f:
            all_responses = json.load(f)
        
        
        l=len(all_responses["captions"])
        
        labels=all_responses['labels']

        # 获取所有类别
        all_classes = coco_object_categories

        # 创建一个字典，将类别映射到索引
        class_to_index = {cls: idx for idx, cls in enumerate(all_classes)}

        # 将标签数据转换为类似one-hot的形式
        def labels_to_onehot(label_list, class_to_index):
            num_classes = len(class_to_index)
            onehot_tensor = torch.zeros(num_classes)

            for label in label_list:
                index = class_to_index[label]
                onehot_tensor[index] = 1

                return onehot_tensor

        # 将所有样本的标签转换为one-hot编码的形式
        onehot_labels = [labels_to_onehot(sample_labels, class_to_index) for sample_labels in labels]
        labels = torch.stack(onehot_labels)

        captions=all_responses["captions"]
        captions=torch.cat([clip.tokenize(p) for p in captions])

        train=[]
        for i in range(60000):
            train.append((captions[i],labels[i]))

        #=======get few-shot=======

        
        super().__init__(train_x=train, val=test[0::10], test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})

    def read_name_list(self, path):
        ret = []
        with open(path, 'r') as f:
            for line in f:
                tmp = line.strip().split(' ')
                ret.append(tmp[0])
        return ret

    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)

        print("Splitting data into 50% train, 20% val, and 30% test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y - 1, classname=c)  # convert to 0-based label
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train : n_train + n_val], label, cname))
            test.extend(_collate(impaths[n_train + n_val :], label, cname))

        return train, val, test
