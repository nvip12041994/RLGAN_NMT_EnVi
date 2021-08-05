#!/bin/bash
TEXT=./pure_data
fairseq-preprocess  --source-lang en --target-lang vi \
                    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
                    --joined-dictionary \
                    --destdir new-data-bin/iwslt15.tokenized.en-vi 