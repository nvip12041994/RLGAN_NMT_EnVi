#!/usr/bin/bash

# --criterion RL \
# --sample-beam 5 \
# --bpe subword_nmt \

ARCH=transformer
export CUDA_VISIBLE_DEVICES=0

DATA_PATH=./new-data-bin/iwslt15.tokenized.en-vi/
MODEL_PATH=./model/


fairseq-train ${DATA_PATH} \
--user-dir /home/s1910443/experiment/change_seq/RLGAN_NMT_EnVi \
--arch $ARCH \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
--dropout 0.3 --weight-decay 0.0001 \
--optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--max-tokens 4096 \
--save-dir $MODEL_PATH \
--seed 2048 \
--restore-file $MODEL_PATH/checkpoint_best.pt