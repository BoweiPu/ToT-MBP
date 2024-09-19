
# Mutil-label task code for Category-Specific Channel Pruning for Effective Modality Bridging in Text-only Tuning


## Install

The code is based largely on the implementation of [CoOp](https://github.com/KaiyangZhou/CoOp), [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) and [TaI-DPT](https://github.com/guozix/TaI-DPT).


Please follow the steps below to build your environment.

```bash
# Create a conda environment (Omit if you already have a suitable environment)
conda create -n dassl python=3.7
conda activate dassl
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge # torch (version >= 1.7.1)

# Clone this repo
git clone https://github.com/guozix/MBP-multi.git
cd MBP-multi

# install Dassl
cd Dassl.pytorch-master/
# Install dependencies
pip install -r requirements.txt
# Install this library (no need to re-build if the source code is modified)
python setup.py develop

cd ..
# Install CLIP dependencies
pip install -r requirements.txt

# Finished
```

## Datasets
We use captions from MS-COCO and localized narratives from OpenImages, and we evaluate our method on VOC2007, MS-COCO and NUS-WIDE.
The directory structure is organized as follows.

The multi-label classification datasets can be accessed from their official websites [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/), [MSCOCO2014](https://cocodataset.org/#download) and [NUS-WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html). The raw images of NUS-WIDE can be accessed from [here](https://pan.baidu.com/s/1Bj-7fdrZAvUJPqAKrUkbbQ) (verification code: s6oj).

For the OpenImages dataset, we only use the localized narratives of its "V6" version. The ~130MB jsonl file can be downloaded from the official [site](https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_captions.jsonl).

```
DATAROOT
├── OpenImages
│   ├── captions
│   │   └── open_images_train_v6_captions.jsonl
├── VOCdevkit
│   ├── VOC2007
|   │   ├── Annotations
|   │   ├── caption_data
|   │   ├── ImageSets
|   │   │   ├── Layout
|   │   │   ├── Main
|   │   │   └── Segmentation
|   │   ├── JPEGImages
|   │   ├── SegmentationClass
|   │   └── SegmentationObject
├── COCO
│   ├── annotations
│   ├── train2014
│   └── val2014
└── NUSWIDE
    ├── ImageList
    │   ├── Imagelist.txt
    │   ├── TestImagelist.txt
    │   └── TrainImagelist.txt
    ├── Flickr
    │   ├── actor
    │   ├── administrative_assistant
    │   ├── adobehouses
    │   ├── adult
    │   ...
    ├── TrainTestLabels
    └── Concepts81.txt
```
<!-- Tai-DPT provide images of NUS-WIDE used in our experiments:
https://github.com/guozix/TaI-DPT -->

## Usage


**Training**

(If you extracted the trained model in the previous evaluation step, you need to set a different `run_ID`, otherwise the script will skip the training.)

Train MBP-multi on the datasets:
``` bash
cd scripts/
bash train.sh end 16 False run_ID

```

## Thanks

We use code from [CoOp](https://github.com/KaiyangZhou/CoOp), [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) and [TaI-DPT](https://github.com/guozix/TaI-DPT), which are great repositories and we encourage you to check them out and cite them in your work.
