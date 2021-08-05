#!/usr/bin/bash

# --criterion RL \
# --sample-beam 5 \

ARCH=transformer
export CUDA_VISIBLE_DEVICES=0

DATA_PATH=./new-data-bin/iwslt15.tokenized.en-vi/
MODEL_PATH=./model/


fairseq-train ${DATA_PATH} \
--user-dir /home/s1910443/experiment/change_seq/RLGAN_NMT_EnVi \
--arch $ARCH \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
--dropout 0.2 \
--optimizer adam --lr 0.005 --lr-shrink 0.5 \
--max-tokens 12000 \
--save-dir $MODEL_PATH \
--seed 2048 \
--bpe subword_nmt \
--restore-file $MODEL_PATH/checkpoint_best.pt