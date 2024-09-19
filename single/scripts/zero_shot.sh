#!/bin/bash
# custom config
DATA=/root/datasets
TRAINER=clip_adapt
CFG=vit_b32
id="$1"
for CFG in  vit_b32  vit_b16  rn50 rn101 rn50x4
do
    for dset in dtd eurosat fgvc_aircraft oxford_flowers ucf101  food101 imagenet_r   sun397
    do
    for seed in  0
        do
        DIR=output/${id}/${TRAINER}/${CFG}/${dset}/${seed}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        CUDA_VISIBLE_DEVICES=0 python main.py \
            --root ${DATA} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/"${dset}".yaml \
            --config-file configs/trainers/adapt/${CFG}.yaml \
            --output-dir  ${DIR}  \
            --txt_epochs 50 \
            --lr 0.0001 \
            --txt_cls 2 \
            --seed ${seed} \
            --zero_shot
    fi
        done
    done
done
