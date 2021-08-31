# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os

from torch.utils.data import ConcatDataset


from data import (
    data_utils, Dictionary, LanguagePairDataset, IndexedInMemoryDataset,
    IndexedRawTextDataset, iterators,
)

def eval_bool(x, default=False):
    if x is None:
        return default
    try:
        return bool(eval(x))
    except TypeError:
        return default
    
class TranslationTask():
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

    # def __init__(self, args):
    #     self.args = args

    @classmethod
    def setup_task(self, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        
        args.left_pad_source = eval_bool(args.left_pad_source)
        args.left_pad_target = eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')
        # print("================================================================")
        # print(args.data)
        # print("================================================================")
        # load dictionaries
        src_dict = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.target_lang)))

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.args = args
        self.datasets = {}
        # print("================================================================")
        # print(args.data)
        # print("================================================================")
        return src_dict, tgt_dict
    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from data import FairseqDataset
        if split not in self.datasets:
            raise KeyError('Dataset not loaded: ' + split)
        if not isinstance(self.datasets[split], FairseqDataset):
            raise TypeError('Datasets are expected to be of type FairseqDataset')
        return self.datasets[split]
    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch.
                Default: ``None``
            max_sentences (int, optional): max number of sentences in each
                batch. Default: ``None``
            max_positions (optional): max sentence length supported by the
                model. Default: ``None``
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long. Default: ``False``
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N. Default: ``1``
            seed (int, optional): seed for random number generator for
                reproducibility. Default: ``1``
            num_shards (int, optional): shard the data iterator into N
                shards. Default: ``1``
            shard_id (int, optional): which shard of the data iterator to
                return. Default: ``0``

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        from data import FairseqDataset
        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        indices = data_utils.filter_by_size(
            indices, dataset.size, max_positions, raise_exception=(not ignore_invalid_inputs),
        )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
        )
        
    def load_dataset(self, split, combine=False):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
       
        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))                    
            if (self.args.raw_text and IndexedRawTextDataset.exists(filename)):                
                return True
            elif ((not self.args.raw_text) and IndexedInMemoryDataset.exists(filename)):                
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedInMemoryDataset.exists(path):
                return IndexedInMemoryDataset(path, fix_lua_indexing=True)
            return None

        src_datasets = []
        tgt_datasets = []

        data_paths = [self.args.data]
        # print("================================================================")
        # print(data_paths)
        # print(self.args.source_lang)
        # print(self.args.target_lang)
        # print("================================================================")
        #print(data_paths)
        for data_path in data_paths:
            # print("================================================================")
            # print(data_path)
            # print("================================================================")
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                # print("================================================================")
                # print(split_k)
                # print("================================================================")
                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

                if not combine:
                    break
        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            src_sizes = src_dataset.sizes
            tgt_sizes = tgt_dataset.sizes
        else:
            if self.args.upsample_primary > 1:
                src_datasets.extend([src_datasets[0]] * (self.args.upsample_primary - 1))
                tgt_datasets.extend([tgt_datasets[0]] * (self.args.upsample_primary - 1))
            src_dataset = ConcatDataset(src_datasets)
            tgt_dataset = ConcatDataset(tgt_datasets)
            src_sizes = np.concatenate([ds.sizes for ds in src_datasets])
            tgt_sizes = np.concatenate([ds.sizes for ds in tgt_datasets])

        self.datasets[split] = LanguagePairDataset(
            src_dataset, src_sizes, self.src_dict,
            tgt_dataset, tgt_sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)
    
    def set_source_dictionary(self, src_dict):
        """set the source :class:`~fairseq.data.Dictionary`."""
        self.src_dict = src_dict
    
    def set_target_dictionary(self, tgt_dict):
        """set the target :class:`~fairseq.data.Dictionary`."""
        self.tgt_dict = tgt_dict
    
    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
