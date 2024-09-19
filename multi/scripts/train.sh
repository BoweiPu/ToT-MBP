#!/bin/bash

cd ..

# custom config
DATA=/root/datasets
TRAINER=MBP

CTP=$1  # class token position (end or middle)
NCTX=$2  # number of context tokens
CSC=$3  # class-specific context (False or True)
run_ID=$4

export CUDA_VISIBLE_DEVICES=2
SEED=2
rate=0.3

for rate in  0.1 0.3 0.5
    do
    for SEED in   1 2 3
        do

        for CFG in rn50  rn101 rn50x4 #vitb32  vitb16  
            do
            for DATASET in voc2007_distill coco2014_distill   nuswide_distill_limit  
                do
                DIR=output/${run_ID}/${rate}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED}
                if [ -d "$DIR" ]; then
                    echo "Results are available in ${DIR}. Skip this job"
                else
                    echo "Run this job andsave the output to ${DIR}"
                    python train_caption.py \
                    --root ${DATA} \
                    --seed ${SEED} \
                    --trainer ${TRAINER} \
                    --dataset-config-file configs/datasets/${DATASET}.yaml \
                    --config-file configs/trainers/${CFG}.yaml \
                    --output-dir ${DIR} \
                    --r ${rate} \
                    TRAINER.Caption.N_CTX ${NCTX} \
                    TRAINER.Caption.CSC ${CSC} \
                    TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP}
                fi
                done
            done
        done
    done
 
