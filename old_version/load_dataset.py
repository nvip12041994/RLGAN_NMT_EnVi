import argparse
import logging

import torch
import os
from torch import cuda
import options
import data

parser = argparse.ArgumentParser(
    description="Driver program for JHU Adversarial-NMT.")

options.add_dataset_args(parser)
def main(args):
    print(args.data)
    dataset = data.load_dataset(
                args.data,
                ['test'],
                args.src_lang,
                args.trg_lang,
            )
    print(vars(dataset))
    src_dict, dst_dict = data.load_dictionaries(
        args.data,
        args.src_lang,
        args.trg_lang,
    )
    print(src_dict)
if __name__ == "__main__":
    ret = parser.parse_known_args()
    args = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(
            parser.parse_known_args()[1]))
    main(args)