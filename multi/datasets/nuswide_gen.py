from calendar import c
import os
from os.path import join
from re import L
from numpy import dtype
import pickle5 as pickle
import random
from scipy.io import loadmat
from collections import defaultdict
import torch
import json
import jsonlines
from tqdm import tqdm
from clip import clip
from clip.model import convert_weights

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

from .data_helpers import *

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

object_categories = nuswide_object_categories
classname_synonyms = nuswide_classname_synonyms

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
class nuswide_gen(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = 'NUSWIDE'
        cls_num = 81
    
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "Flickr")
        self.cls_name_list = self.read_name_list(join(self.dataset_dir, 'Concepts81.txt'), False)
        self.im_name_list = self.read_name_list(join(self.dataset_dir, 'ImageList/TrainImagelist.txt'), False)
        self.im_name_list_test = self.read_name_list(join(self.dataset_dir, 'ImageList/TestImagelist.txt'), False)
        print('NUS-WIDE total {} images, test {} images. '.format(len(self.im_name_list), len(self.im_name_list_test)))

        json_name_all=f"captions/nuswide.json"
        with open(json_name_all, 'r') as f:
            all_responses = json.load(f)
        
        
        l=len(all_responses["captions"])
        
        labels=all_responses['labels']

        # 获取所有类别
        all_classes = nuswide_object_categories

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
        ############################
        ## test data

        path_labels = os.path.join(self.dataset_dir, 'TrainTestLabels')
        # labeled_data = dict()
        num_classes = len(object_categories)

        test_labels = [] # torch.ones((len(self.im_name_list_test), num_classes))
        for i in tqdm(range(num_classes)):
            file_ = os.path.join(path_labels, 'Labels_'+object_categories[i]+'_Test.txt')
            cls_labels = []
            with open(file_, 'r') as f:
                for j, line in enumerate(f):
                    tmp = line.strip()
                    cls_labels.append(int(tmp))
            test_labels.append(torch.tensor(cls_labels, dtype=torch.long))
        test_labels = torch.stack(test_labels, dim=1)
        
        test = []
        for i, name in enumerate(self.im_name_list_test):
            item_ = Datum(impath=self.image_dir + '/' + '/'.join(name.split('\\')), label=test_labels[i], classname='')
            test.append(item_)

        super().__init__(train_x=train, val=test[0::20], test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})

    def read_name_list(self, path, if_split=True):
        ret = []
        with open(path, 'r') as f:
            for line in f:
                if if_split:
                    tmp = line.strip().split(' ')
                    ret.append(tmp[0])
                else:
                    tmp = line.strip()
                    ret.append(tmp)
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
