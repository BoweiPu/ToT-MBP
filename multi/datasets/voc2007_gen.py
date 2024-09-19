from calendar import c
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

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

from .data_helpers import *

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from captions.data_helpers import *
object_categories = voc_object_categories
classname_synonyms = voc_classname_synonyms

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
class VOC2007_gen(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = 'VOCdevkit/VOC2007'
        cls_num = len(object_categories)
    
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "JPEGImages")



        json_name_all=f"captions/voc2007.json"
        with open(json_name_all, 'r') as f:
            all_responses = json.load(f)
        
        
        l=len(all_responses["captions"])
        
        labels=all_responses['labels']

        # 获取所有类别
        all_classes = voc_object_categories

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
        for i in range(l):
            train.append((captions[i],labels[i]))
        ############################################################################################################
        ## test data
        test_data_imname2label = read_object_labels(self.dataset_dir, phase='test')
        self.im_name_list_test = read_im_name_list(join(self.dataset_dir, 'ImageSets/Main/test.txt'))
    
        test = []
        for name in self.im_name_list_test:
            item_ = Datum(impath=self.image_dir+'/{}.jpg'.format(name), label=test_data_imname2label[name], classname='')
            test.append(item_)

        super().__init__(train_x=train, val=test, test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})
