import argparse
import logging

import torch
import os
from torch import cuda
import options
import data
from generator import LSTMModel, LSTMEncoder, LSTMDecoder
import utils
from sequence_generator import SequenceGenerator
from torch.serialization import default_restore_location

print("funny")
parser = argparse.ArgumentParser(
    description="Adversarial-NMT.")
options.add_distributed_training_args(parser)
options.add_dataset_args(parser)
path="checkpoints/generator/best_gmodel.pt"
def main(args):
    use_cuda = (len(args.gpuid) >= 1)
    if args.gpuid:
        cuda.set_device(args.gpuid[0])
    dataset = data.load_dataset(
                args.data,
                ['test'],
                args.src_lang,
                args.trg_lang,
            )
    src_dict, dst_dict = data.load_dictionaries(
        args.data,
        args.src_lang,
        args.trg_lang,
    )
    print('| [{}] dictionary: {} types'.format(
            dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(
        dataset.dst, len(dataset.dst_dict)))
    print('| {} {} {} examples'.format(
        args.data, 'test', len(dataset.splits['test'])))

    print('| loading model(s) from {}'.format(path))
    #generator = LSTMModel(LSTMEncoder,LSTMDecoder)
    state = torch.load(path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    state = utils._upgrade_state_dict(state)



if __name__ == "__main__":
    ret = parser.parse_known_args()
    args = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(
            parser.parse_known_args()[1]))
    main(args)