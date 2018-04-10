#!/usr/bin/env python

import sys

from cvangysel import argparse_utils, logging_utils, \
    multiprocessing_utils, trec_utils

import argparse
import logging
import os
import operator
import pyndri
import pyndri.utils

import nvsm


class RankFn(object,
             metaclass=multiprocessing_utils.WorkerMetaclass):

    @staticmethod
    def worker(payload):
        topic_id, topic_token_ids = payload

        if topic_token_ids is None:
            return topic_id, None

        logging.info('Query %s: %s',
                     topic_id, topic_token_ids)

        topic_repr = RankFn.model.query_representation(topic_token_ids)
        if topic_repr is None:
            return topic_id, None

        kwargs = {
            'similarity_fn': 'cosine',
        }

        if isinstance(RankFn.args.top_k, int):
            kwargs['results_requested'] = RankFn.args.top_k
        elif isinstance(RankFn.args.top_k, dict):
            if topic_id not in RankFn.args.top_k:
                logging.warning('Skipping topic %s as there are '
                                'no judged documents.',
                                topic_id)

                return topic_id, None

            kwargs['results_requested'] = len(RankFn.args.top_k[topic_id])

            kwargs['document_set'] = set(map(
                operator.itemgetter(1),
                RankFn.args.index.document_ids(RankFn.args.top_k[topic_id])))
        else:
            raise RuntimeError()

        topic_scores_and_documents = RankFn.model.query(
            topic_token_ids, **kwargs)

        if not topic_scores_and_documents:
            return topic_id, None

        topic_scores_and_documents = [
            (-score, RankFn.args.index.document(int_doc_id)[0])
            for score, int_doc_id in topic_scores_and_documents]

        return topic_id, (topic_repr, topic_scores_and_documents)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--loglevel', type=str, default='INFO')

    parser.add_argument('--num_workers',
                        type=argparse_utils.positive_int, default=16)

    parser.add_argument('--topics', nargs='+',
                        type=argparse_utils.existing_file_path)

    parser.add_argument('model', type=argparse_utils.existing_file_path)

    parser.add_argument('--index', required=True)

    parser.add_argument('--linear', action='store_true', default=False)
    parser.add_argument('--self_information',
                        action='store_true',
                        default=False)
    parser.add_argument('--l2norm_phrase', action='store_true', default=False)

    parser.add_argument('--bias_coefficient',
                        type=argparse_utils.ratio,
                        default=0.0)

    parser.add_argument('--rerank_exact_matching_documents',
                        action='store_true',
                        default=False)

    parser.add_argument('--strict', action='store_true', default=False)

    parser.add_argument('--top_k', default=None)

    parser.add_argument('--num_queries',
                        type=argparse_utils.positive_int,
                        default=None)

    parser.add_argument('run_out')

    args = parser.parse_args()

    args.index = pyndri.Index(args.index)

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    if not args.top_k:
        args.top_k = 1000
    elif args.top_k == 'all':
        args.top_k = args.top_k = \
            args.index.maximum_document() - args.index.document_base()
    elif args.top_k.isdigit():
        args.top_k = int(args.top_k)
    elif all(map(os.path.exists, args.top_k.split())):
        topics_and_documents = {}

        for qrel_path in args.top_k.split():
            with open(qrel_path, 'r') as f_qrel:
                for topic_id, judgments in trec_utils.parse_qrel(f_qrel):
                    if topic_id not in topics_and_documents:
                        topics_and_documents[topic_id] = set()

                    for doc_id, _ in judgments:
                        topics_and_documents[topic_id].add(doc_id)

        args.top_k = topics_and_documents
    else:
        raise RuntimeError()

    logging.info('Loading dictionary.')
    dictionary = pyndri.extract_dictionary(args.index)

    logging.info('Loading model.')
    model_base, epoch_and_ext = args.model.rsplit('_', 1)
    epoch = int(epoch_and_ext.split('.')[0])

    if not os.path.exists('{}_meta'.format(model_base)):
        model_meta_base, batch_idx = model_base.rsplit('_', 1)
    else:
        model_meta_base = model_base

    kwargs = {
        'strict': args.strict,
    }

    if args.self_information:
        kwargs['self_information'] = True

    if args.linear:
        kwargs['bias_coefficient'] = args.bias_coefficient
        kwargs['nonlinearity'] = None

    if args.l2norm_phrase:
        kwargs['l2norm_phrase'] = True

    model = nvsm.load_model(
        nvsm.load_meta(model_meta_base),
        model_base, epoch, **kwargs)

    for topic_path in args.topics:
        run_out_path = '{}-{}'.format(
            args.run_out, os.path.basename(topic_path))

        if os.path.exists(run_out_path):
            logging.warning('Run for topics %s already exists (%s); skipping.',
                            topic_path, run_out_path)

            continue

        queries = list(pyndri.utils.parse_queries(
            args.index, dictionary, topic_path,
            strict=args.strict,
            num_queries=args.num_queries))

        if args.rerank_exact_matching_documents:
            assert not isinstance(args.top_k, dict)

            topics_and_documents = {}

            query_env = pyndri.TFIDFQueryEnvironment(args.index)

            for topic_id, topic_token_ids in queries:
                topics_and_documents[topic_id] = set()

                query_str = ' '.join(
                    dictionary[term_id] for term_id in topic_token_ids
                    if term_id is not None)

                for int_doc_id, score in query_env.query(
                        query_str, results_requested=1000):
                    topics_and_documents[topic_id].add(
                        args.index.ext_document_id(int_doc_id))

            args.top_k = topics_and_documents

        run = trec_utils.OnlineTRECRun(
            'cuNVSM', rank_cutoff=(
                args.top_k if isinstance(args.top_k, int)
                else sys.maxsize))

        rank_fn = RankFn(
            args.num_workers,
            args=args, model=model)

        for idx, (topic_id, topic_data) in enumerate(rank_fn(queries)):
            if topic_data is None:
                continue

            logging.info('Query %s (%d/%d)', topic_id, idx + 1, len(queries))

            (topic_repr,
             topic_scores_and_documents) = topic_data

            run.add_ranking(topic_id, topic_scores_and_documents)

            del topic_scores_and_documents

        run.close_and_write(run_out_path, overwrite=False)

        logging.info('Run outputted to %s.', run_out_path)

    del rank_fn

if __name__ == '__main__':
    sys.exit(main())
