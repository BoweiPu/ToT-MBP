#!/bin/bash
# custom config
DATA=/root/datasets
TRAINER=clip_adapt
id="$1"

for CFG in vit_b32  vit_b16  rn50 rn101 rn50x4
do
   for seed in  1 2 3 4 5 6
        do
for rate in 0.5 1  2 3  4  5  6 7 8 9 10
 do

    for dset in  oxford_flowers eurosat dtd fgvc_aircraft  ucf101  imagenet_r food101   sun397
    do
 
       DIR=output/${id}/${rate}/${CFG}/${dset}/${seed}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        CUDA_VISIBLE_DEVICES=1 python main.py \
            --root ${DATA} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/"${dset}".yaml \
            --config-file configs/trainers/adapt/${CFG}.yaml \
            --output-dir  ${DIR}  \
            --txt_epochs 50 \
            --lr 0.0001 \
            --txt_cls 2 \
            --seed ${seed} \
            --rate ${rate} #\
            #--t ${t}
    fi
        done
    done
done
done