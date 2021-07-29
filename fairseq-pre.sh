#!/bin/bash
TEXT=./data
fairseq-preprocess  --source-lang en --target-lang vi \
                    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
                    --destdir new-data-bin/iwslt15.tokenized.en-vi 