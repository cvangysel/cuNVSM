#!/usr/bin/env python

import sys

from cvangysel import argparse_utils, logging_utils

import argparse
import logging
import os
import pyndri

import nvsm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--loglevel', type=str, default='INFO')

    parser.add_argument('--index',
                        type=argparse_utils.existing_directory_path,
                        required=True)
    parser.add_argument('--model',
                        type=argparse_utils.existing_file_path,
                        required=True)
    parser.add_argument('--vocabulary_list',
                        type=argparse_utils.nonexisting_file_path,
                        required=True)

    args = parser.parse_args()

    args.index = pyndri.Index(args.index)

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    logging.info('Loading dictionary.')
    dictionary = pyndri.extract_dictionary(args.index)

    logging.info('Loading model.')
    model_base, epoch_and_ext = args.model.rsplit('_', 1)
    epoch = int(epoch_and_ext.split('.')[0])

    if not os.path.exists('{}_meta'.format(model_base)):
        model_meta_base, batch_idx = model_base.rsplit('_', 1)
    else:
        model_meta_base = model_base

    model = nvsm.load_model(
        nvsm.load_meta(model_meta_base),
        model_base, epoch)

    with open(args.vocabulary_list, 'w') as f_vocabulary_list:
        for index_term_id in model.term_mapping:
            f_vocabulary_list.write(dictionary[index_term_id])
            f_vocabulary_list.write('\n')

if __name__ == '__main__':
    sys.exit(main())
