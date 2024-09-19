# Single-label task code for Category-Specific Channel Pruning for Effective Modality Bridging in Text-only Tuning


This is the official repository for our paper *TAP: TARGETED PROMPTING FOR TASK ADAPTIVE GENERATION OF TEXTUAL
TRAINING INSTANCES FOR VISUAL CLASSIFICATION*. We provide the code for reproducing the results 
for all the 8 datasets used in our paper.

## Installation

Our code is built upon the official codebase of the [CoOp](https://github.dev/KaiyangZhou/CoOp) paper and has been 
tested in an environment having `python 3.8.8` and `pytorch 2.0.1` compiled with `CUDA 11.6`. 

As a first step, install `dassl` library (under `single/`) in your environment by following the instructions [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation).

To further install all other dependencies, please run the following command, after having your environment activated:

```pip install -r requirements.txt```

## Datasets

Please download and structure your datasets according to the instructions provided in the [CoOp](https://github.dev/KaiyangZhou/CoOp)
official repository. All the `8` datasets should be present in the `data/` directory.

## Descriptions

The generic and dataset specific descriptions for all the 8 datasets are present in the `descriptions/` directory.

## Experiments

### MBP
To reproduce the results for `MBP` all the 8 datasets in Table 1, please run the following command:

```bash scripts/train.sh runID```


### Zero-Shot
Similarly, to obtain zero-shot CLIP results with the single prompt template `a photo of a {category}`. Please run: 

```bash scripts/zeroshot.sh runID```



