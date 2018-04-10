import h5py
import heapq
import itertools
import numpy as np
import logging
import scipy.spatial.distance
import sklearn.neighbors

from cvangysel import sklearn_utils
from nvsm_pb2 import Metadata


def load_meta(path):
    meta = Metadata()

    with open('{}_meta'.format(path), 'rb') as f_meta:
        meta.ParseFromString(f_meta.read())

    return meta


def load_model(meta, path, epoch, **kwargs):
    with h5py.File('{}_{}.hdf5'.format(path, epoch), 'r') as f_model:
        return LSE(meta, f_model, **kwargs)


class NearestNeighbors(object):

    """Wrapper around sklearn.neighbors.NearestNeighbors
       that is optimized for cosine distance."""

    def __init__(self, metric='cosine', **kwargs):
        self.metric = metric

        nn_metric = 'euclidean' if self.metric == 'cosine' else self.metric

        if 'algorithm' not in kwargs:
            kwargs['algorithm'] = sklearn_utils.neighbors_algorithm(nn_metric)

        logging.info('Using %s algorithm for nearest neighbor retrieval '
                     'using %s metric.',
                     kwargs['algorithm'], nn_metric)

        self.nn_impl = sklearn.neighbors.NearestNeighbors(
            metric=nn_metric, **kwargs)

    def fit(self, X, *args, **kwargs):
        if self.metric == 'cosine':
            X = X.copy()
            X /= scipy.linalg.norm(X, axis=1, keepdims=True)

        result = self.nn_impl.fit(X, *args, **kwargs)

        logging.info('Data was fitted using %s method.',
                     self.nn_impl._fit_method)

        return result

    def kneighbors(self, X, *args, inplace=False, **kwargs):
        if self.metric == 'cosine':
            if not inplace:
                X = X.copy()

            X /= scipy.linalg.norm(X, axis=1, keepdims=True)

        result = self.nn_impl.kneighbors(X, *args, **kwargs)

        if kwargs.get('return_distance', True):
            dist, ind = result

            if self.metric == 'cosine':
                # Euclidean distance and cosine similarity/distance are related
                # as follows.
                #
                # First note that cosine distance is equal to 1.0 - cos(A, B).
                # Consequently, cosine distance operates in the domain (0, 2)
                # where higher values indicates dissimilarity. Cosine
                # distance can be converted into cosine similarity using
                # the formula:
                #
                #   -cos_distance + 1.0.
                #
                # If A and B are normalized, then Euclidean distance and cosine
                # similarity are related as follows:
                #
                #   ||A - B||^2 = 2 * cos_distance
                #   cos_distance = ||A - B||^2 / 2
                #
                # Note that x^2 / 2 is a monotonically increasing function when
                # x is positive.
                #
                # Consequently, sorting according to x or according to x^2/2
                # results in the same ranking (for positive x).
                #
                # Given that Euclidean distance is always positive (due to
                # being a metric), we rely on it as a metric during nearest
                # neighbor search.

                dist = np.power(dist, 2.0) / 2.0

            return dist, ind
        else:
            return result


class TermBruteforcer(object):

    def __init__(self, model, max_ngram_cardinality=1):
        self.model = model

        reprs = []

        for k in range(1, max_ngram_cardinality + 1):
            logging.info('Computing %d-gram indices.', k)

            combination_idx = np.array(
                list(itertools.combinations(
                    range(model.word_representations.shape[0]), k)),
                dtype=np.int32)\
                .reshape(-1)

            logging.info('Obtaining %d-gram phrase representations.', k)

            phrase_repr = model.word_representations[combination_idx]\
                .reshape(-1, k, model.word_representations.shape[1])\
                .mean(axis=1)

            logging.info('Computing %d-gram projections.', k)

            phrase_projection = model.infer(phrase_repr)

            reprs.append(phrase_projection)

        logging.info('Indexing k-NN.')

        self.projection_neighbors = NearestNeighbors(
            metric='cosine',
            n_neighbors=20)

        self.projection_neighbors.fit(np.vstack(reprs))

    def search(self, projected_query_repr):
        if projected_query_repr is None:
            return None

        projected_query_repr = projected_query_repr.copy()

        if projected_query_repr.ndim < 2:
            projected_query_repr = projected_query_repr.reshape(1, -1)

        neighbor_weights, neighbor_idx = \
            self.projection_neighbors.kneighbors(projected_query_repr)

        neighbor_weights = - neighbor_weights + 1

        nearby_ngrams = [[
            (self.model.inv_term_mapping[word_idx],
             neighbor_weights[0, idx])
            for idx, word_idx in enumerate(neighbor_idx[f_idx, :])]
            for f_idx in range(projected_query_repr.shape[0])]

        return nearby_ngrams


class NVSM(object):

    def __init__(self, meta, f_model,
                 only_word_embeddings=False,
                 only_object_embeddings=False,
                 self_information=False,
                 bias_coefficient=0.0,
                 nonlinearity=np.tanh,
                 strict=False):
        self.total_terms = meta.total_terms

        self.self_information = self_information
        self.nonlinearity = nonlinearity

        self.strict = strict

        if not only_object_embeddings:
            self.word_representations = \
                f_model['word_representations-representations'][()]

            self.num_terms = self.word_representations.shape[0]
            self.term_repr_size = self.word_representations.shape[1]

            self.term_mapping = {}
            self.inv_term_mapping = {}

            self.inv_term_id_to_term_freq = {}

            for term in meta.term:
                assert term.index_term_id not in self.term_mapping
                assert term.model_term_id < self.num_terms
                self.term_mapping[term.index_term_id] = term.model_term_id

                assert term.model_term_id not in self.inv_term_mapping
                self.inv_term_mapping[term.model_term_id] = term.index_term_id

                assert term.model_term_id not in self.inv_term_id_to_term_freq
                self.inv_term_id_to_term_freq[term.model_term_id] = \
                    term.term_frequency

        if not only_word_embeddings:
            self.object_representations = \
                f_model['entity_representations-representations'][()]

            self.num_objects = self.object_representations.shape[0]
            self.object_repr_size = self.object_representations.shape[1]

            self.object_mapping = {}
            self.inv_object_mapping = {}

            for o in meta.object:
                assert o.model_object_id not in self.object_mapping
                assert o.model_object_id < self.object_representations.shape[0]

                self.object_mapping[o.model_object_id] = o.index_object_id

                assert o.index_object_id not in self.inv_object_mapping
                self.inv_object_mapping[o.index_object_id] = o.model_object_id

        if not only_word_embeddings and not only_object_embeddings:
            self.transform_matrix = \
                f_model['word_entity_mapping-transform'][()]

            if not bias_coefficient != 0.0:
                self.transform_bias = (
                    bias_coefficient *
                    f_model['word_entity_mapping-bias'][()].ravel())
            else:
                self.transform_bias = None

            assert (self.term_repr_size, self.object_repr_size) == \
                self.transform_matrix.shape

            if self.transform_bias is not None:
                assert (self.object_repr_size,) == \
                    self.transform_bias.shape

    def __repr__(self):
        return '<NVSM with {} words ({}-dimensional) and ' \
               '{} entities ({}-dimensional).'.format(
                   self.num_terms, self.term_repr_size,
                   self.num_objects, self.object_repr_size)

    def get_average_object_repr(self):
        if not hasattr(self, 'average_obj_repr'):
            self.average_obj_repr = np.mean(
                self.object_representations,
                axis=0)

        return self.average_obj_repr

    def get_average_word_repr(self):
        if not hasattr(self, 'average_word_repr'):
            self.average_word_repr = np.mean(
                self.word_representations,
                axis=0)

        return self.average_word_repr

    def get_word_repr(self, index_term_id):
        if index_term_id not in self.term_mapping:
            logging.warning('Term %s is out of vocabulary.',
                            index_term_id)

            return None

        return self.word_representations[
            self.term_mapping[index_term_id], :]

    def query_representation(self, index_term_ids):
        model_terms = []

        for index_term_id in index_term_ids:
            if index_term_id not in self.term_mapping:
                if self.strict:
                    logging.debug('Term %s is out of vocabulary; '
                                  'skipping query.',
                                  index_term_id)

                else:
                    logging.debug('Term %s is out of vocabulary; '
                                  'skipping term.',
                                  index_term_id)

                continue

            model_terms.append(self.term_mapping[index_term_id])

        if not model_terms or (
                self.strict and len(model_terms) < len(index_term_ids)):
            return None

        if self.self_information:
            model_term_weights = [
                -np.log(self.inv_term_id_to_term_freq[model_term] /
                        self.total_terms)
                for model_term in model_terms]
        else:
            model_term_weights = None

        average_term_repr = np.average(
            self.word_representations[model_terms, :],
            axis=0, weights=model_term_weights)

        return average_term_repr

    def infer(self, query_repr):
        if query_repr is None:
            return None

        projected_term_repr = np.dot(query_repr, self.transform_matrix)

        if self.transform_bias is not None:
            projected_term_repr += self.transform_bias

        if self.nonlinearity is not None:
            projected_term_repr = self.nonlinearity(projected_term_repr)

        return projected_term_repr

    def related_terms(self, index_term_id):
        if index_term_id not in self.term_mapping:
            logging.warning('Term %s is out of vocabulary.',
                            index_term_id)

            return None

        if not hasattr(self, 'word_neighbors'):
            self.word_neighbors = NearestNeighbors(
                metric='cosine',
                n_neighbors=30)
            self.word_neighbors.fit(self.word_representations)

        nearest = self.word_neighbors.kneighbors(self.word_representations[
            self.term_mapping[index_term_id], :])

        return [self.inv_term_mapping[model_term_id]
                for model_term_id in nearest[1][0, :].tolist()]

    def term_similarity(self, first_index_term_id, second_index_term_id):
        if first_index_term_id not in self.term_mapping or \
                second_index_term_id not in self.term_mapping:
            return None

        return 1.0 - scipy.spatial.distance.cosine(
            self.word_representations[
                self.term_mapping[first_index_term_id], :],
            self.word_representations[
                self.term_mapping[second_index_term_id], :])

    def query(self, index_terms, *args, **kwargs):
        projected_term_repr = self.infer(
            self.query_representation(index_terms))

        return self.query_using_projected_query(
            projected_term_repr, *args, **kwargs)

    def query_using_projected_query(
            self, projected_term_repr,
            similarity_fn='cosine',
            similarity_fn_include_prior=False,
            results_requested=1000,
            document_set=None):
        if projected_term_repr is None:
            return None

        if document_set:
            document_set = set(document_set)

        assert projected_term_repr.size == self.object_repr_size, \
            projected_term_repr.shape

        projected_term_repr = projected_term_repr.ravel().reshape(1, -1)

        results_requested = min(
            results_requested,
            self.object_representations.shape[0])

        if (not similarity_fn_include_prior and
                results_requested is not None and
                document_set is None):
            if not hasattr(self, 'object_neighbors'):
                self.object_neighbors = NearestNeighbors(
                    metric=similarity_fn,
                    n_neighbors=results_requested)
                self.object_neighbors.fit(self.object_representations)

                self.query_similarity_fn = similarity_fn

            assert self.query_similarity_fn == similarity_fn

            nearest_dist, nearest_ind = self.object_neighbors.kneighbors(
                projected_term_repr,
                return_distance=True,
                n_neighbors=results_requested)

            topic_scores_and_documents = [
                (nearest_dist[0, rank],
                 self.object_mapping[nearest_ind[0, rank]])
                for rank in range(nearest_ind.size)]
        else:
            if isinstance(similarity_fn, str):
                actual_similarity_fn = getattr(scipy.spatial.distance,
                                               similarity_fn)

                def similarity_fn(first, second, int_obj_id):
                    return actual_similarity_fn(first, second)

            iterable = (
                (float(
                    similarity_fn(
                        projected_term_repr,
                        self.object_representations[object_idx, :],
                        int_obj_id=self.object_mapping[object_idx])),
                 self.object_mapping[object_idx])
                for object_idx in range(self.object_representations.shape[0])
                if not document_set or
                self.object_mapping[object_idx] in document_set)

            if results_requested is not None:
                topic_scores_and_documents = heapq.nsmallest(
                    n=results_requested, iterable=iterable)
            else:
                topic_scores_and_documents = sorted(iterable)

        return topic_scores_and_documents

    def score_documents(self, index_term_ids, int_document_ids):
        projected_term_repr = self.infer(
            self.query_representation(index_term_ids))

        if projected_term_repr is None:
            return

        assert projected_term_repr.shape == (1, self.object_repr_size)

        for document_id in int_document_ids:
            if document_id not in self.inv_object_mapping:
                continue

            similarity = 1.0 - scipy.spatial.distance.cosine(
                projected_term_repr,
                self.object_representations[
                    self.inv_object_mapping[document_id], :])

            yield document_id, similarity

LSE = NVSM  # Backwards compatibility.
