#!/usr/bin/env python

import sys

from cvangysel import argparse_utils, logging_utils, trec_utils

import argparse
import collections
import html.parser
import logging
import operator


class ReutersParser(html.parser.HTMLParser):

    def __init__(self):
        super(ReutersParser, self).__init__()

        self.documents = []

        self.__current_text_tag = None
        self.__current_category_tag = None

    def handle_starttag(self, tag, attrs):
        if tag == 'reuters':
            self.documents.append({
                'doc_id': str(len(self.documents)),
                'texts': {},
                'tags': collections.defaultdict(set),
            })
        elif tag in {'title', 'dateline', 'body'}:
            self.__current_text_tag = tag
        elif tag in {'topics', 'places', 'companies', 'orgs', 'exchanges'}:
            self.__current_category_tag = tag

    def handle_endtag(self, tag):
        if tag == self.__current_text_tag:
            self.__current_text_tag = None

        if tag == self.__current_category_tag:
            self.__current_category_tag = None

    def handle_data(self, data):
        assert not (
            (self.__current_text_tag is not None) and
            (self.__current_category_tag is not None)), (
                self.__current_text_tag, self.__current_category_tag)

        if self.__current_text_tag is not None:
            self.documents[-1]['texts'][self.__current_text_tag] = data

        if self.__current_category_tag is not None:
            self.documents[-1]['tags'][self.__current_category_tag].add(data)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--loglevel', type=str, default='INFO')

    parser.add_argument('--shard_size',
                        type=argparse_utils.positive_int, default=1000000)

    parser.add_argument('sgm',
                        type=argparse_utils.existing_file_path,
                        nargs='+')

    parser.add_argument('--top_k_topics',
                        type=argparse_utils.positive_int,
                        default=20)

    parser.add_argument('--trectext_out_prefix', type=str, required=True)

    parser.add_argument('--document_classification_out',
                        type=argparse_utils.nonexisting_file_path,
                        required=True)

    args = parser.parse_args()

    try:
        logging_utils.configure_logging(args)
    except IOError:
        return -1

    parser = ReutersParser()

    for sgm_path in args.sgm:
        logging.info('Parsing %s.', sgm_path)

        with open(sgm_path, 'r', encoding='ISO-8859-1') as f_sgm:
            parser.feed(f_sgm.read())

    logging.info('Parsed %d documents.', len(parser.documents))

    topic_histogram = collections.Counter(
        topic
        for document in parser.documents
        for topic in document['tags']['topics'])

    top_topics = set(sorted(
        topic_histogram.keys(),
        key=lambda topic: topic_histogram[topic])[-args.top_k_topics:])

    logging.info('Top topics: %s', top_topics)

    writer = trec_utils.ShardedTRECTextWriter(
        args.trectext_out_prefix,
        shard_size=args.shard_size, encoding='latin1')

    with open(args.document_classification_out, 'w') as \
            f_document_classification_out:
        for document in parser.documents:
            doc_id = document['doc_id']

            doc_text = '\n'.join([document['texts'].get('title', ''),
                                  document['texts'].get('dateline', ''),
                                  document['texts'].get('body', '')])

            writer.write_document(doc_id, doc_text)

            doc_topics = {
                topic
                for topic in document['tags']['topics']
                if topic in top_topics}

            if doc_topics:
                most_specific_doc_topic = min(
                    doc_topics, key=lambda topic: topic_histogram[topic])

                f_document_classification_out.write(doc_id)
                f_document_classification_out.write(' ')
                f_document_classification_out.write(most_specific_doc_topic)
                f_document_classification_out.write('\n')

    writer.close()

if __name__ == '__main__':
    sys.exit(main())
