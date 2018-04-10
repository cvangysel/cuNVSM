#!/usr/bin/env python

import sys

from cvangysel import argparse_utils, logging_utils

import argparse
import logging
import matplotlib.cm as cm
import matplotlib.markers as markers
import matplotlib.pyplot as plt
import numpy as np
import os
import pylatex.utils
import pyndri
from sklearn.manifold import TSNE

import nvsm

MARKERS = ['o', 's', '<', '>', '^', 'v', 'd', 'p', '*', '8',
           '1', '2', '3', '4',
           markers.TICKLEFT, markers.TICKRIGHT,
           markers.TICKUP, markers.TICKDOWN,
           markers.CARETLEFT, markers.CARETRIGHT,
           markers.CARETUP, markers.CARETDOWN]

plt.rcParams["figure.figsize"] = (8.0, 4.25)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model')

    parser.add_argument('index', type=argparse_utils.existing_directory_path)

    parser.add_argument('--limit',
                        type=argparse_utils.positive_int,
                        default=None)

    parser.add_argument('--object_classification',
                        type=argparse_utils.existing_file_path,
                        nargs='+',
                        default=None)

    parser.add_argument('--filter_unclassified',
                        action='store_true',
                        default=False)

    parser.add_argument('--l2_normalize',
                        action='store_true',
                        default=False)

    parser.add_argument('--mode',
                        choices=('tsne', 'embedding_projector'),
                        default='tsne')

    parser.add_argument('--legend',
                        action='store_true',
                        default=False)

    parser.add_argument('--tick_labels',
                        action='store_true',
                        default=False)

    parser.add_argument('--edges',
                        action='store_true',
                        default=False)

    parser.add_argument('--border',
                        action='store_true',
                        default=False)

    parser.add_argument('--plot_out',
                        type=argparse_utils.nonexisting_file_path,
                        required=True)

    args = parser.parse_args()

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    # Set matplotlib style.
    plt.style.use('bmh')

    logging.info('Loading index.')
    index = pyndri.Index(args.index)

    logging.info('Loading cuNVSM model.')
    model_base, epoch_and_ext = args.model.rsplit('_', 1)
    epoch = int(epoch_and_ext.split('.')[0])

    if not os.path.exists('{}_meta'.format(model_base)):
        model_meta_base, batch_idx = model_base.rsplit('_', 1)
    else:
        model_meta_base = model_base

    model = nvsm.load_model(
        nvsm.load_meta(model_meta_base),
        model_base, epoch,
        only_object_embeddings=True)

    raw_object_representations = np.copy(model.object_representations)

    if args.limit:
        raw_object_representations = raw_object_representations[:args.limit, :]

    for object_classification in args.object_classification:
        root, ext = os.path.splitext(args.plot_out)

        plot_out = '{}-{}.{}'.format(
            root, os.path.basename(object_classification), ext.lstrip('.'))

        if object_classification and args.filter_unclassified:
            logging.info('Filtering unclassified.')

            with open(object_classification, 'r') as f_objects:
                object_ids = [line.strip().split()[0] for line in f_objects]
                indices = sorted(model.inv_object_mapping[idx]
                                 for _, idx in index.document_ids(object_ids)
                                 if idx in model.inv_object_mapping)

                logging.info('Considering %d out of %d representations.',
                             len(indices), len(object_ids))

                translation_table = {idx: i for i, idx in enumerate(indices)}

                object_representations = raw_object_representations[indices]

                assert object_representations.shape[0] == \
                    len(translation_table)
        else:
            translation_table = None

            raise NotImplementedError()

        logging.info('Loading object clusters.')

        cluster_id_to_product_ids = {}

        if object_classification:
            with open(object_classification, 'r') as f_objects:
                for line in f_objects:
                    object_id, cluster_id = line.strip().split()

                    if cluster_id not in cluster_id_to_product_ids:
                        cluster_id_to_product_ids[cluster_id] = set()

                    cluster_id_to_product_ids[cluster_id].add(object_id)

                for cluster_id in list(cluster_id_to_product_ids.keys()):
                    object_ids = list(cluster_id_to_product_ids[cluster_id])

                    cluster_id_to_product_ids[cluster_id] = set(
                        (model.inv_object_mapping[int_object_id]
                            if translation_table is None
                            else translation_table[
                                model.inv_object_mapping[int_object_id]])
                        for ext_object_id, int_object_id in
                        index.document_ids(object_ids)
                        if int_object_id in model.inv_object_mapping and
                        (args.limit is None or
                         (model.inv_object_mapping[int_object_id] <
                             args.limit)))
        else:
            raise NotImplementedError()

        assert len(cluster_id_to_product_ids) < len(MARKERS)

        if args.l2_normalize:
            logging.info('L2-normalizing representations.')

            object_representations /= np.linalg.norm(
                object_representations,
                axis=1, keepdims=True)

        if args.mode == 'tsne':
            logging.info('Running t-SNE.')

            twodim_object_representations = \
                TSNE(n_components=2, init='pca', random_state=0).\
                fit_transform(object_representations)

            logging.info('Plotting %s.', twodim_object_representations.shape)

            colors = cm.rainbow(
                np.linspace(0, 1, len(cluster_id_to_product_ids)))

            for idx, cluster_id in enumerate(
                    sorted(cluster_id_to_product_ids.keys(),
                           key=lambda cluster_id: len(
                               cluster_id_to_product_ids[cluster_id]),
                           reverse=True)):
                row_ids = list(cluster_id_to_product_ids[cluster_id])

                plt.scatter(
                    twodim_object_representations[row_ids, 0],
                    twodim_object_representations[row_ids, 1],
                    marker=MARKERS[idx],
                    edgecolors='grey' if args.edges else None,
                    cmap=plt.cm.Spectral,
                    color=colors[idx],
                    alpha=0.3,
                    label=pylatex.utils.escape_latex(cluster_id))

            plt.grid()

            plt.tight_layout()

            if args.legend:
                plt.legend(bbox_to_anchor=(0, -0.15, 1, 0),
                           loc=2,
                           ncol=2,
                           mode='expand',
                           borderaxespad=0)

            if not args.tick_labels:
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)

            if not args.border:
                # plt.gcf().patch.set_visible(False)
                plt.gca().axis('off')

            logging.info('Writing %s.', plot_out)

            plt.savefig(plot_out,
                        bbox_inches='tight',
                        transparent=True,
                        pad_inches=0,
                        dpi=200)
        elif args.mode == 'embedding_projector':
            logging.info('Dumping to TensorFlow embedding projector format.')

            with open('{}_vectors.tsv'.format(plot_out), 'w') as f_vectors, \
                    open('{}_meta.tsv'.format(plot_out), 'w') as f_meta:
                f_meta.write('document_id\tclass\n')

                def write_rowids(row_ids, cluster_id):
                    for row_id in row_ids:
                        f_vectors.write(
                            '{}\n'.format('\t'.join(
                                '{:.5f}'.format(x)
                                for x in object_representations[row_id])))

                        f_meta.write('{}\t{}\n'.format(
                            index.ext_document_id(
                                model.object_mapping[row_id]),
                            cluster_id))

                for cluster_id in cluster_id_to_product_ids.keys():
                    row_ids = list(cluster_id_to_product_ids[cluster_id])
                    write_rowids(row_ids, cluster_id)

    logging.info('All done!')

if __name__ == '__main__':
    sys.exit(main())
