import collections
import itertools
import os
import math
import torch

from translation import TranslationTask
from models import generator
import argparse
import utils

parser = argparse.ArgumentParser(description="Adversarial-NMT.")
def load_dataset_splits(task, splits):
    for split in splits:
        print(split)
        if split == 'train':
            task.load_dataset(task, split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(task, split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e

def main(args):
    # if args.max_tokens is None:
    #     args.max_tokens = 6000
    # print(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    
    print(" Setup task, e.g., translation, language modeling, etc.")
    task = TranslationTask()
    task.setup_task(args)
    
    print("Load dataset splits")
    load_dataset_splits(TranslationTask, ['train', 'valid'])
    # Build model
    model = generator.TransformerModel.build_model(args,task)
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))
    
    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )
    dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens, max_positions)
    
    # Initialize dataloader
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset('train'),
        max_tokens=args.max_tokens,
        max_sentences=None,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
    )
    

if __name__ == '__main__':
    ret = parser.parse_known_args()
    options = ret[0]    
    generator.transformer_iwslt_de_en(options)
    options.device_id = getattr(options, 'device_id', 0)
    options.seed = getattr(options, 'seed', 12345)
    options.data = getattr(options, 'data', "./v060_data-bin/iwslt15.tokenized.en-vi")
    options.source_lang = getattr(options, 'source_lang', 'en')
    options.target_lang = getattr(options, 'target_lang', 'vi')
    options.left_pad_source = getattr(options, 'left_pad_source', 'True')
    options.left_pad_target = getattr(options, 'left_pad_target', 'False')
    options.raw_text = getattr(options, 'raw_text',False)
    options.max_source_positions = getattr(options, 'max_source_positions', 1024)
    options.max_target_positions = getattr(options, 'max_target_positions', 1024)
    options.max_tokens = getattr(options, 'max_tokens', 1024)
    
    main(options)