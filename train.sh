#!/usr/bin/bash
data=./new-data-bin/iwslt15.tokenized.en-vi/
fairseq-train ${data} --task translation --user-dir /home/s1910443/experiment/change_seq/RLGAN_NMT_EnVi \
    --arch gru_transformer --encoder-layers 2 --decoder-layers 2 --dropout 0.3 --share-all-embeddings \
    --criterion RL \
    --sample-beam 5 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --log-format 'simple' --log-interval 100 --save-dir "checkpoints/${VOCAB}" \
    --batch-size 100 --max-update 100000 --update-freq 2