#!/usr/bin/bash
mkdir -p joint
for file in checkpoints/joint/*; do
    echo $(basename "$file")
    name=$(basename "$file")
    folder_name=${name/.pt/ }
    mkdir -p joint/$folder_name
    python3 generate.py --model_file $(basename "$file") --data new-data-bin/iwslt15.tokenized.en-vi/ --src_lang en --trg_lang vi --batch-size 16 --gpuid 0
    mv real.txt joint/$folder_name
    mv predictions.txt joint/$folder_name
    perl scripts/multi-bleu.perl joint/$folder_name/real.txt < joint/$folder_name/predictions.txt
done