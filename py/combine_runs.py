#!/usr/bin/env python

import sys

from cvangysel import argparse_utils, logging_utils, trec_utils

import argparse
import collections
import logging
import pytrec_eval
import sklearn.model_selection
import numpy as np


def compute_combined_run(runs, weights, query_ids, normalizer_impl):
    combined_run = {}

    for query_id in query_ids:
        combined_ranking = collections.defaultdict(list)

        for run_idx, run in enumerate(runs):
            ranking = run[query_id]

            normalizer = normalizer_impl(list(ranking.values()))

            for object_id, score in ranking.items():
                combined_ranking[object_id].append(
                    weights[run_idx] * normalizer(score))

        combined_run[query_id] = {
            object_id: np.mean(scores)
            for object_id, scores in combined_ranking.items()}

    return combined_run


class StandardizationNormalizer(object):

    def __init__(self, scores):
        self.mean = np.mean(scores)
        self.std = np.std(scores)

    def __call__(self, score):
        return (score - self.mean) / self.std


class MinMaxNormalizer(object):

    def __init__(self, scores):
        self.min = np.min(scores)
        self.max = np.max(scores)

    def __call__(self, score):
        return (score - self.min) / (self.max - self.min)


class IdentityNormalizer(object):

    def __init__(self, scores):
        pass

    def __call__(self, score):
        return score


SCORE_NORMALIZERS = {
    'standardize': StandardizationNormalizer,
    'minmax': MinMaxNormalizer,
    'none': IdentityNormalizer,
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--loglevel', type=str, default='INFO')

    # Supervised learning (direct optimization).
    parser.add_argument('--qrel',
                        type=argparse_utils.existing_file_path,
                        default=None)
    parser.add_argument('--num_folds',
                        type=argparse_utils.positive_int,
                        default=20)
    parser.add_argument('--alpha_stepsize',
                        type=argparse_utils.ratio,
                        default=0.05)

    # Unsupervised learning.
    parser.add_argument('--alpha',
                        type=argparse_utils.ratio,
                        default=None)

    parser.add_argument('--runs', nargs=2,
                        type=argparse_utils.existing_file_path)

    parser.add_argument('--score_normalizer',
                        choices=SCORE_NORMALIZERS,
                        required=True)

    parser.add_argument('--measure',
                        choices=pytrec_eval.supported_measures,
                        default='map_cut_1000')

    parser.add_argument('run_out',
                        type=argparse_utils.nonexisting_file_path)

    args = parser.parse_args()

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    assert (args.qrel is None) != (args.alpha is None)

    runs = []

    for run_path in args.runs:
        with open(run_path, 'r') as f_run:
            runs.append(pytrec_eval.parse_run(f_run))

    run = trec_utils.OnlineTRECRun('combined')

    if args.qrel is not None:
        with open(args.qrel, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map_cut'})

        query_ids = list(qrel.keys())
        np.random.shuffle(query_ids)

        kfold = sklearn.model_selection.KFold(n_splits=args.num_folds)

        for fold_idx, (train_query_indices, test_query_indices) in \
                enumerate(kfold.split(query_ids)):
            def _generate():
                for alpha in np.arange(0.0, 1.0, args.alpha_stepsize):
                    weights = [alpha, 1.0 - alpha]

                    assert len(runs) == len(weights)

                    combined_run = compute_combined_run(
                        runs, weights,
                        [query_ids[idx] for idx in train_query_indices],
                        normalizer_impl=SCORE_NORMALIZERS[
                            args.score_normalizer])

                    results = evaluator.evaluate(combined_run)

                    agg_measure_value = pytrec_eval.compute_aggregated_measure(
                        args.measure,
                        [query_measures[args.measure]
                         for query_measures in results.values()])

                    yield agg_measure_value, alpha

            best_measure_value, best_alpha = max(_generate())

            logging.info('Fold %d: best_alpha=%.2f train %s=%.4f',
                         fold_idx, best_alpha,
                         args.measure, best_measure_value)

            test_combined_run = compute_combined_run(
                runs,
                [best_alpha, 1.0 - best_alpha],
                [query_ids[idx] for idx in test_query_indices],
                normalizer_impl=SCORE_NORMALIZERS[
                    args.score_normalizer])

            for query_id, document_ids_and_scores in test_combined_run.items():
                run.add_ranking(
                    query_id,
                    [(score, document_id)
                     for document_id, score in
                     document_ids_and_scores.items()])
    elif args.alpha is not None:
        query_ids = sorted(set.union(*[set(run.keys()) for run in runs]))

        test_combined_run = compute_combined_run(
            runs,
            [args.alpha, 1.0 - args.alpha],
            query_ids,
            normalizer_impl=SCORE_NORMALIZERS[args.score_normalizer])

        for query_id, document_ids_and_scores in test_combined_run.items():
            run.add_ranking(
                query_id,
                [(score, document_id)
                 for document_id, score in
                 document_ids_and_scores.items()])
    else:
        raise NotImplementedError()

    run.close_and_write(args.run_out, overwrite=False)

    logging.info('Run outputted to %s.', args.run_out)

if __name__ == '__main__':
    sys.exit(main())
