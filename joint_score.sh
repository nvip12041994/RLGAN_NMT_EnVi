#!/usr/bin/bash
mkdir -p joint
for file in checkpoints/joint/*; do
    echo $(basename "$file")
    name=$(basename "$file")
    folder_name=${name/.pt/ }
    mkdir -p joint/$folder_name
    python3 joint_generate.py --model_file checkpoints/joint/$(basename "$file") --data new-data-bin/iwslt15.tokenized.en-vi/ --src_lang en --trg_lang vi --batch-size 16 --gpuid 0
    perl scripts/multi-bleu.perl real.txt < predictions.txt
    mv -v real.txt joint/$folder_name/
    mv -v predictions.txt joint/$folder_name/
done