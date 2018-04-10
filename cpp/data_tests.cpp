#include <gflags/gflags.h>

using namespace __gnu_cxx;
#include <indri/QueryEnvironment.hpp>
#include <indri/DiskIndex.hpp>

#include <indri/MemoryIndex.hpp>
#include <indri/MemoryIndexVocabularyIterator.hpp>

#include "cuNVSM/tests_base.h"
#include "cuNVSM/data.h"

DEFINE_string(test_data_dir, "", "Directory pointing to test data.");

using ::testing::Eq;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Pair;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::UnorderedElementsAre;

namespace indri {
index::DiskIndex* LoadIndex(const std::string& repository_path);
}  // namespace indri

template <typename ParentClass>
class DataSourceTestHelper : public ParentClass {
 protected:
  std::vector<WordIdxType> get_features(const TextEntity::Batch& batch) {
      return std::vector<WordIdxType>(
          batch.features_,
          &batch.features_[batch.num_instances_ * batch.window_size()]);
  }

  std::vector<WeightType> get_feature_weights(const TextEntity::Batch& batch) {
      return std::vector<WeightType>(
          batch.feature_weights_,
          &batch.feature_weights_[batch.num_instances_ * batch.window_size()]);
  }

  std::vector<ObjectIdxType> get_labels(const TextEntity::Batch& batch) {
      return std::vector<ObjectIdxType>(
          batch.labels_,
          &batch.labels_[batch.num_instances_]);
  }

  template <typename BatchT>
  std::vector<WeightType> get_weights(const BatchT& batch) {
      return std::vector<WeightType>(
          batch.weights_,
          &batch.weights_[batch.num_instances_]);
  }

  std::vector<std::pair<ObjectIdxType, ObjectIdxType>> get_features(const RepresentationSimilarity::Batch& batch) {
      std::vector<std::pair<ObjectIdxType, ObjectIdxType>> features;

      for (size_t idx = 0; idx < batch.num_instances_; ++idx) {
          features.push_back(
              std::make_pair(batch.features_[2 * idx],
                             batch.features_[2 * idx + 1]));
      }

      return features;
  }
};

namespace TextEntity {

TEST(InMemoryDocumentSource, InMemoryDocumentSource) {
    const VocabularyT vocabulary = construct_vocabulary(
        {"hello", "world", "freedom",
         "of", "speech", "does",
         "not", "exist", "or",
         "to", "that", "is", "question"});

    InMemoryDocumentSource source(
        vocabulary,
        {
            {1L, "hello world freedom of speech"},
            {2L, "speech freedom does not exist"},
            {3L, "exist or not to exist that is question"},
        });

    Batch batch(1024 /* batch_size */, 3 /* window_size */);
    source.next(&batch);

    EXPECT_EQ(batch.num_instances(), 12);
}

TEST(InMemoryDocumentSource, pad_batch) {
    const VocabularyT vocabulary = construct_vocabulary(
        {"hello", "world", "freedom",
         "of", "speech", "does",
         "not", "exist", "or",
         "to", "that", "is", "question"});

    InMemoryDocumentSource source(
        vocabulary,
        {
            {1L, "hello world freedom of speech"},
            {2L, "speech freedom does not exist"},
            {3L, "exist or not to exist that is question"},
        },
        true /* pad_batch */);

    Batch batch(1024 /* batch_size */, 3 /* window_size */);
    source.next(&batch);

    EXPECT_EQ(batch.num_instances(), 1024);
}

class NullSource : public DataSource {
 public:
  NullSource()
      : DataSource(11 /* vocabulary_size */,
                   11 /* corpus_size */) {
      reset();
  }

  virtual void reset() {}

 private:
  FRIEND_TEST(DataSourceTest, create_instances);
  FRIEND_TEST(DataSourceTest, create_instances_overflow);
};

typedef DataSourceTestHelper<::testing::Test> DataSourceTest;

TEST_F(DataSourceTest, create_instances) {
    NullSource source;

    Batch batch(6, /* batch_size */
                3 /* window_size */);

    source.create_instances<std::vector<size_t>>(
        {1, 2, 3, 4, 5, 6, 7, 8},
        1337 /* object_id */,
        1.0 /* weight */,
        1 /* stride */,
        &batch);

    EXPECT_THAT(get_features(batch),
                ElementsAreArray({
                    1, 2, 3,
                    2, 3, 4,
                    3, 4, 5,
                    4, 5, 6,
                    5, 6, 7,
                    6, 7, 8}));
}

TEST_F(DataSourceTest, create_instances_overflow) {
    NullSource source;

    Batch batch(2, /* batch_size */
                3 /* window_size */);

    source.create_instances<std::vector<size_t>>(
        {1, 2, 3, 4, 5, 6, 7, 8},
        1337 /* object_id */,
        1.0 /* weight */,
        1 /* stride */,
        &batch);
    EXPECT_TRUE(source.has_next());

    EXPECT_THAT(get_features(batch),
                ElementsAre(
                    1, 2, 3,
                    2, 3, 4));

    batch.clear();
    EXPECT_TRUE(source.has_next());
    source.next(&batch);
    EXPECT_THAT(get_features(batch),
                ElementsAre(
                    3, 4, 5,
                    4, 5, 6));

    batch.clear();
    EXPECT_TRUE(source.has_next());
    source.next(&batch);
    EXPECT_THAT(get_features(batch),
                ElementsAre(
                    5, 6, 7,
                    6, 7, 8));

    batch.clear();
    EXPECT_FALSE(source.has_next());
}

class MockDiskIndex : public ::indri::index::DiskIndex {
 public:
  MOCK_METHOD0(documentMaximum, lemur::api::DOCID_T());
  MOCK_METHOD1(documentLength, int (lemur::api::DOCID_T));
  MOCK_METHOD1(termList, const ::indri::index::TermList*(lemur::api::DOCID_T));

  MOCK_METHOD0(vocabularyIterator, ::indri::index::VocabularyIterator*());

  MOCK_METHOD0(uniqueTermCount, UINT64());
  MOCK_METHOD0(termCount, UINT64());
  MOCK_METHOD0(documentCount, UINT64());
};

class MockVocabularyIterator : public ::indri::index::VocabularyIterator {
 public:
  explicit MockVocabularyIterator(const std::vector<::indri::index::DiskTermData*>& items)
      : items_(items) {}

  void startIteration() {
      it_ = items_.begin();
  }

  ::indri::index::DiskTermData* currentEntry() {
      CHECK(it_ != items_.end());

      return *it_;
  }

  bool nextEntry() {
      if (it_ == items_.end()) {
          return false;
      }

      ++it_;
      return true;
  }

  bool nextEntry(const char *skipTo) {
      return nextEntry();
  }

  bool finished() {
      return it_ == items_.end();
  }

 private:
  std::vector<::indri::index::DiskTermData*> items_;
  std::vector<::indri::index::DiskTermData*>::iterator it_;
};

class IndriSourceTest : public DataSourceTest {
 protected:
  IndriSourceTest() : rng_(new RNG) {}

  ::indri::index::DiskIndex* create_index(const bool add_oov = false, const bool actual_tf = false) const {
      MockDiskIndex* const index = new MockDiskIndex;

      EXPECT_CALL(*index, documentMaximum())
          .WillRepeatedly(Return(2));

      EXPECT_CALL(*index, termCount())
          .WillRepeatedly(Return(10000));
      EXPECT_CALL(*index, documentCount())
          .WillRepeatedly(Return(2));

      ::indri::index::VocabularyIterator* const vocabulary_it =
          new MockVocabularyIterator({
              create_term(1, actual_tf ? 2 : 5),
              create_term(2, actual_tf ? 3 : 5),
              create_term(3, actual_tf ? 2 : 5),
              create_term(4, actual_tf ? 1 : 5),
              create_term(5, actual_tf ? 1 : 5),
              create_term(10, actual_tf ? 1 : 5),
              create_term(111, actual_tf ? 1 : 5),
          });

      EXPECT_CALL(*index, vocabularyIterator())
          .WillOnce(Return(vocabulary_it));
      EXPECT_CALL(*index, uniqueTermCount())
          .WillRepeatedly(Return(7));

      EXPECT_CALL(*index, documentLength(0))
          .WillRepeatedly(Return(7 + (add_oov ? 3 : 0)));

      ::indri::index::TermList* const term_list_d0 = new ::indri::index::TermList;
      term_list_d0->addTerm(1);
      term_list_d0->addTerm(2);
      term_list_d0->addTerm(3);
      term_list_d0->addTerm(4);

      if (add_oov) {
          term_list_d0->addTerm(0);
          term_list_d0->addTerm(0);
          term_list_d0->addTerm(0);
      }

      term_list_d0->addTerm(3);
      term_list_d0->addTerm(2);
      term_list_d0->addTerm(1);

      EXPECT_CALL(*index, termList(0))
          .WillOnce(Return(term_list_d0));

      EXPECT_CALL(*index, documentLength(1))
          .WillRepeatedly(Return(4 + (add_oov ? 5 : 0)));

      ::indri::index::TermList* const term_list_d1 = new ::indri::index::TermList;
      term_list_d1->addTerm(10);
      term_list_d1->addTerm(2);

      if (add_oov) {
          term_list_d1->addTerm(0);
          term_list_d1->addTerm(0);
          term_list_d1->addTerm(0);
          term_list_d1->addTerm(0);
          term_list_d1->addTerm(0);
      }

      term_list_d1->addTerm(111);
      term_list_d1->addTerm(5);

      EXPECT_CALL(*index, termList(1))
          .WillOnce(Return(term_list_d1));

      return index;
  }

  ::indri::index::DiskTermData* create_term(const int32 term_id, const int32 total_count) const {
      ::indri::index::DiskTermData* term_entry = new ::indri::index::DiskTermData;

      term_entry->termID = term_id;
      term_entry->termData = new ::indri::index::TermData;
      term_entry->termData->term = "test";
      term_entry->termData->corpus = ::indri::index::TermFieldStatistics();
      term_entry->termData->corpus.totalCount = total_count;

      return term_entry;
  }

  void get_instances(IndriSource* const source,
                     std::vector<std::pair<ObjectIdxType, std::vector<WordIdxType>>>* const instances) {
      instances->clear();

      Batch batch(4, /* batch_size */
                  source->window_size_);

      while (source->has_next()) {
          source->next(&batch);

          const std::vector<WordIdxType> batch_features = get_features(batch);
          const std::vector<ObjectIdxType> batch_labels = get_labels(batch);

          CHECK_EQ(batch_features.size() % batch_labels.size(), 0);

          for (size_t i = 0; i < batch_labels.size(); ++i) {
              instances->push_back(std::make_pair(
                  batch_labels[i],
                  std::vector<WordIdxType>(
                      batch_features.begin() + (i * source->window_size_),
                      batch_features.begin() + ((i + 1) * source->window_size_))));
          }

          batch.clear();
      }
  }

  std::unique_ptr<RNG> rng_;
};

TEST_F(IndriSourceTest, IndriSource) {
    IndriSource source(
        create_index(true /* add_oov */),
        3, /* window_size */
        rng_.get(),
        0, /* max_vocabulary_size */
        0, /* min_document_frequency */
        0, /* max_document_frequency */
        0, /* documents_cutoff */
        false, /* include_oov */
        false, /* include_digits */
        false, /* shuffle */
        NONE /* sampling_strategy */);

    EXPECT_EQ(source.vocabulary_size(), 7);
    EXPECT_EQ(source.corpus_size(), 2);

    // Document #0: 1 2 3 4 3 2 1 -> 0 1 2 3 2 1 0
    // Document #1: 10 2 111 5 -> 5 1 6 4

    EXPECT_THAT(source.term_id_mapping(),
                UnorderedElementsAre(
                    Pair(1, 0),
                    Pair(2, 1),
                    Pair(3, 2),
                    Pair(4, 3),
                    Pair(5, 4),
                    Pair(10, 5),
                    Pair(111, 6)));

    EXPECT_THAT(source.document_id_mapping(),
                UnorderedElementsAre(
                    Pair(0, 0),
                    Pair(1, 1)));

    Batch batch(4, /* batch_size */
                3 /* window_size */);

    // Document #0.
    EXPECT_TRUE(source.has_next());
    source.next(&batch);

    EXPECT_THAT(get_features(batch),
                ElementsAreArray({
                    0, 1, 2,
                    1, 2, 3,
                    2, 3, 2,
                    3, 2, 1 /* Document #0 */}));

    EXPECT_THAT(get_feature_weights(batch),
                ElementsAreArray({
                    FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0),
                    FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0),
                    FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0),
                    FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0)}));

    EXPECT_THAT(get_labels(batch),
                ElementsAre(0, 0, 0, 0 /* Document #0 */));

    const float32 avg_doc_length = 9.5;

    EXPECT_THAT(get_weights(batch),
                ElementsAre(FPHelper<FloatT>::eq(avg_doc_length / 10.0),
                            FPHelper<FloatT>::eq(avg_doc_length / 10.0),
                            FPHelper<FloatT>::eq(avg_doc_length / 10.0),
                            FPHelper<FloatT>::eq(avg_doc_length / 10.0) /* Document #0 */));

    batch.clear();

    // Document #1.
    EXPECT_TRUE(source.has_next());
    source.next(&batch);

    EXPECT_THAT(get_features(batch),
                ElementsAre(
                    2, 1, 0, /* Document #0 */
                    5, 1, 6,
                    1, 6, 4 /* Document #1 */));

    EXPECT_THAT(get_feature_weights(batch),
                ElementsAreArray({
                    FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0),
                    FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0),
                    FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0), FPHelper<FloatT>::eq(1.0)}));

    EXPECT_THAT(get_labels(batch),
                ElementsAre(0, /* Document #0 */
                            1, 1 /* Document #1 */));

    EXPECT_THAT(get_weights(batch),
                ElementsAre(FPHelper<FloatT>::eq(avg_doc_length / 10.0), /* Document #0 */
                            FPHelper<FloatT>::eq(avg_doc_length / 9.0),
                            FPHelper<FloatT>::eq(avg_doc_length / 9.0) /* Document #1 */));

    EXPECT_FALSE(source.has_next());

    // Verify if reset works.
    source.reset();
    EXPECT_TRUE(source.has_next());
}

TEST_F(IndriSourceTest, IndriSource_UnsupportedSampling_Death) {
    EXPECT_DEATH(
        IndriSource source(
            create_index(),
            3, /* window_size */
            rng_.get(),
            0, /* max_vocabulary_size */
            0, /* min_document_frequency */
            0, /* max_document_frequency */
            0, /* documents_cutoff */
            false, /* include_oov */
            false, /* include_digits */
            false, /* shuffle */
            NGRAM_FREQUENCY /* sampling_strategy */),
        "");
}

TEST_F(IndriSourceTest, StochasticIndriSource) {
    IndriSource source(
        create_index(true /* add_oov */),
        3, /* window_size */
        rng_.get(),
        0, /* max_vocabulary_size */
        0, /* min_document_frequency */
        0, /* max_document_frequency */
        0, /* documents_cutoff */
        false, /* include_oov */
        false, /* include_digits */
        true, /* shuffle */
        NONE /* sampling_strategy */);

    std::vector<std::pair<ObjectIdxType, std::vector<WordIdxType>>> instances;
    get_instances(&source, &instances);

    EXPECT_THAT(
        instances,
        UnorderedElementsAre(
            Pair(0, ElementsAre(0, 1, 2)),
            Pair(0, ElementsAre(1, 2, 3)),
            Pair(0, ElementsAre(2, 3, 2)),
            Pair(0, ElementsAre(3, 2, 1)),
            Pair(0, ElementsAre(2, 1, 0)),
            Pair(1, ElementsAre(5, 1, 6)),
            Pair(1, ElementsAre(1, 6, 4))));
}

TEST_F(IndriSourceTest, StochasticIndriSource_Resampling) {
    IndriSource source(
        create_index(true /* add_oov */),
        3, /* window_size */
        rng_.get(),
        0, /* max_vocabulary_size */
        0, /* min_document_frequency */
        0, /* max_document_frequency */
        0, /* documents_cutoff */
        false, /* include_oov */
        false, /* include_digits */
        true, /* shuffle */
        NGRAM_FREQUENCY /* sampling_strategy */);

    std::vector<std::pair<ObjectIdxType, std::vector<WordIdxType>>> instances;
    get_instances(&source, &instances);

    // Relies on seed == 1.
    EXPECT_THAT(
        instances,
        UnorderedElementsAre(
            Pair(0, ElementsAre(0, 1, 2)),
            Pair(0, ElementsAre(0, 1, 2)),
            Pair(0, ElementsAre(2, 3, 2)),
            Pair(0, ElementsAre(3, 2, 1)),
            Pair(1, ElementsAre(5, 1, 6)),
            Pair(1, ElementsAre(5, 1, 6)),
            Pair(1, ElementsAre(1, 6, 4)),
            Pair(1, ElementsAre(1, 6, 4))));
}

TEST_F(IndriSourceTest, StochasticIndriSource_SelfInformation) {
    IndriSource source(
        create_index(true, /* add_oov */
                     true /* actual_tf */),
        3, /* window_size */
        rng_.get(),
        0, /* max_vocabulary_size */
        0, /* min_document_frequency */
        0, /* max_document_frequency */
        0, /* documents_cutoff */
        false, /* include_oov */
        false, /* include_digits */
        true, /* shuffle */
        NGRAM_FREQUENCY, /* sampling_strategy */
        UNIFORM,
        SELF_INFORMATION_TERM_WEIGHTING);

    EXPECT_THAT(source.inv_term_id_to_term_freq_,
                UnorderedElementsAre(
                    Pair(0, 1),
                    Pair(1, 1),
                    Pair(2, 1),
                    Pair(3, 1),
                    Pair(4, 2),
                    Pair(5, 2),
                    Pair(6, 3)));

    Batch batch(4, /* batch_size */
                3 /* window_size */);

    // Document #0.
    EXPECT_TRUE(source.has_next());
    source.next(&batch);

    EXPECT_THAT(get_features(batch),
                ElementsAreArray({
                    6, 3, 1, 5,
                    0, 5, 6, 3,
                    1, 4, 6, 5}));

    EXPECT_THAT(get_feature_weights(batch),
                ElementsAreArray({
                    FPHelper<FloatT>::eq(-log(3.0 / 11.0)),
                    FPHelper<FloatT>::eq(-log(1.0 / 11.0)),
                    FPHelper<FloatT>::eq(-log(1.0 / 11.0)),
                    FPHelper<FloatT>::eq(-log(2.0 / 11.0)),
                    FPHelper<FloatT>::eq(-log(1.0 / 11.0)),
                    FPHelper<FloatT>::eq(-log(2.0 / 11.0)),
                    FPHelper<FloatT>::eq(-log(3.0 / 11.0)),
                    FPHelper<FloatT>::eq(-log(1.0 / 11.0)),
                    FPHelper<FloatT>::eq(-log(1.0 / 11.0)),
                    FPHelper<FloatT>::eq(-log(2.0 / 11.0)),
                    FPHelper<FloatT>::eq(-log(3.0 / 11.0)),
                    FPHelper<FloatT>::eq(-log(2.0 / 11.0))}));
}

TEST_F(IndriSourceTest, document_list) {
    const std::string repository_path = FLAGS_test_data_dir + "/Brown_index";

    const std::vector<std::string> document_list =
        {"cj36", "ck17", "cn04", "cg62", "cm02"};

    IndriSource source(repository_path,
                       3, /* window_size */
                       rng_.get(),
                       0, /* max_vocabulary_size */
                       0, /* min_document_frequency */
                       0, /* max_document_frequency */
                       0, /* documents_cutoff */
                       false, /* include_oov */
                       false, /* include_digits */
                       &document_list);

    EXPECT_EQ(source.corpus_size(), 5);

    EXPECT_THAT(source.document_id_mapping(),
                UnorderedElementsAre(
                    Pair(0, 330),
                    Pair(1, 391),
                    Pair(2, 437),
                    Pair(3, 251),
                    Pair(4, 429)));
}

TEST_F(IndriSourceTest, Brown) {
    const std::string repository_path = FLAGS_test_data_dir + "/Brown_index";

    IndriSource source(repository_path,
                       16, /* window_size */
                       rng_.get(),
                       0, /* max_vocabulary_size */
                       0, /* min_document_frequency */
                       0, /* max_document_frequency */
                       0, /* documents_cutoff */
                       false, /* include_oov */
                       false, /* include_digits */
                       nullptr, /* document_list */
                       nullptr, /* term_blacklist */
                       true, /* shuffle */
                       AUTOMATIC_SAMPLING,
                       UNIFORM);

    EXPECT_EQ(source.corpus_size(), 500);

    size_t i = 0;
    for (const auto& pair : source.document_id_mapping()) {
        CHECK_EQ(std::get<0>(pair), i);
        CHECK_EQ(std::get<1>(pair), i + 1);

        ++i;
    }

    Batch batch(4, /* batch_size */
                16 /* window_size */);

    // Document #0.
    EXPECT_TRUE(source.has_next());
    source.next(&batch);

    std::map<size_t, std::string> str_instances;

    for (size_t i = 0; i < batch.num_instances_; ++ i) {
        std::string tmp = "";
        for (size_t j = 0; j < batch.window_size_; ++j) {
            tmp += source.term(batch.features_[i * batch.window_size_ + j]) + " ";
        }

        insert_or_die(
            source.document_id_mapping().at(batch.labels_[i]),
            tmp,
            &str_instances);
    }

    EXPECT_THAT(
        str_instances,
        UnorderedElementsAre(
            Pair(405, std::string("kept signal dowl car coming steady clear start back hamburger shut device want hang eat dont ")),
            Pair(215, std::string("conspire uncle make secret gift money mother story end child illness delirium brought feverish compulsion ride ")),
            Pair(434, std::string("morgan im usually strong woman im awfully tired hungry start meal eat meal girl cry morgan ")),
            Pair(392, std::string("write pleasant note beautiful last life part ways goodbye forever word fifty dollar add postscript beg "))));
}

}  // namespace TextEntity

namespace RepresentationSimilarity {

typedef DataSourceTestHelper<::testing::Test> RepresentationSimilarityTest;

TEST_F(RepresentationSimilarityTest, LoadSimilarities) {
    const std::map<std::string, ObjectIdxType> identifiers_map {
        {"apple", 0},
        {"pen", 1},
        {"pineapple", 2},
        {"apple-pen", 3},
        {"pineapple-pen", 4},
        {"pen-pineapple-apple-pen", 5},
    };

    std::istringstream stream(std::string(
        "pen apple 1.0\n"
        "pen apple-pen 1.0\n"
        "apple apple-pen 1.0\n"
        "pen pineapple-pen 1.0\n"
        "pineapple pineapple-pen 1.0\n"
        "apple-pen pen-pineapple-apple-pen 1.0\n"
        "pineapple-pen pen-pineapple-apple-pen 1.0\n"
    ));

    std::unique_ptr<std::vector<InstanceT>> data(
        LoadSimilarities(stream, identifiers_map));

    EXPECT_THAT(*data,
                UnorderedElementsAre(
                    make_tuple(1, 0, 1.0),
                    make_tuple(1, 3, 1.0),
                    make_tuple(0, 3, 1.0),
                    make_tuple(1, 4, 1.0),
                    make_tuple(2, 4, 1.0),
                    make_tuple(3, 5, 1.0),
                    make_tuple(4, 5, 1.0)));
}

TEST_F(RepresentationSimilarityTest, DataSource) {
    RNG rng;

    const std::vector<InstanceT>* data = new std::vector<InstanceT>({
        make_tuple(0, 10, 1.0),
        make_tuple(2, 1, 1.0),
        make_tuple(5, 20, 1.0),
        make_tuple(1, 6, 1.0),
        make_tuple(12, 9, 1.0)
    });

    DataSource source(data, &rng);

    Batch batch(1024 /* batch_size */);
    source.next(&batch);

    EXPECT_EQ(batch.num_instances(), 5);

    EXPECT_THAT(get_features(batch),
                UnorderedElementsAre(
                    Pair(0, 10),
                    Pair(2, 1),
                    Pair(5, 20),
                    Pair(1, 6),
                    Pair(12, 9)));
}

}  // namespace RepresentationSimilarity

class CountingSource : public TextEntity::DataSource {
 public:
  CountingSource(const size_t num_batches)
      : TextEntity::DataSource(num_batches /* vocabulary_size */,
                               num_batches /* corpus_size */),
        num_batches_(num_batches),
        batch_idx_(0) {
      reset();
  }

  virtual void reset() override {
      batch_idx_ = 0;
  }

  virtual void next(TextEntity::Batch* const batch) override {
      TextEntity::DataSource::next(batch);

      for (size_t instance_idx = 0; instance_idx < batch->maximum_size(); ++instance_idx) {
          push_instance(std::vector<WordIdxType>(batch->window_size(), batch_idx_), /* tokens */
                        std::vector<WeightType>(), /* token_weights */
                        batch_idx_, /* object_id */
                        1.0, /* weight */
                        batch);
      }

      ++batch_idx_;
  }

  virtual bool has_next() const override {
      return batch_idx_ < num_batches_;
  }

 private:
  const size_t num_batches_;

  size_t batch_idx_;
};

class MetaSourceTest : public ::DataSourceTestHelper<::testing::TestWithParam<int>> {};

INSTANTIATE_TEST_CASE_P(Entropy,
                        MetaSourceTest,
                        ::testing::Range<int>(0 /* start, inclusive */,
                                              11 /* end, exclusive */,
                                              1 /* step */));

TEST_P(MetaSourceTest, AsyncSource) {
    AsyncSource<TextEntity::Batch> source(
        3, /* num_concurrent_batches */
        128, /* batch_size */
        3, /* window_size */
        new CountingSource(8));
    TextEntity::Batch batch(128, /* batch_size*/
                            3 /* window_size */);

    size_t idx = 0;
    while (source.has_next()) {
        source.next(&batch);

        EXPECT_THAT(get_features(batch),
                    ElementsAreArray(std::vector<WordIdxType>(128 * 3, idx)));

        EXPECT_THAT(get_labels(batch),
                    ElementsAreArray(std::vector<WordIdxType>(128, idx)));

        EXPECT_THAT(get_weights(batch),
                    ElementsAreArray(std::vector<WordIdxType>(128, 1.0)));

        batch.clear();

        ++idx;
    }

    CHECK_EQ(idx, 8);

    source.reset();
}

TEST_P(MetaSourceTest, MultiSource) {
    MultiSource<TextEntity::Batch, TextEntity::Batch> source(
        std::make_tuple<DataSource<TextEntity::Batch>*,
                        DataSource<TextEntity::Batch>*>(
            new CountingSource(8), new CountingSource(9)));
    TextEntity::Batch first_batch(128, /* batch_size*/
                                  3 /* window_size */);

    TextEntity::Batch second_batch(32, /* batch_size*/
                                   10 /* window_size */);

    std::tuple<size_t, size_t> args = std::make_tuple<size_t, size_t>(128, 3);

    std::tuple<TextEntity::Batch, TextEntity::Batch> batch(
        std::move(first_batch),
        std::move(second_batch)
    );

    size_t idx = 0;
    while (source.has_next()) {
        source.next(&batch);

        EXPECT_THAT(get_features(std::get<0>(batch)),
                    ElementsAreArray(std::vector<WordIdxType>(128 * 3, idx)));

        EXPECT_THAT(get_labels(std::get<0>(batch)),
                    ElementsAreArray(std::vector<WordIdxType>(128, idx)));

        EXPECT_THAT(get_weights(std::get<0>(batch)),
                    ElementsAreArray(std::vector<WordIdxType>(128, 1.0)));

        EXPECT_THAT(get_features(std::get<1>(batch)),
                    ElementsAreArray(std::vector<WordIdxType>(32 * 10, idx)));

        EXPECT_THAT(get_labels(std::get<1>(batch)),
                    ElementsAreArray(std::vector<WordIdxType>(32, idx)));

        EXPECT_THAT(get_weights(std::get<1>(batch)),
                    ElementsAreArray(std::vector<WordIdxType>(32, 1.0)));

        std::get<0>(batch).clear();
        std::get<1>(batch).clear();

        ++idx;
    }

    CHECK_EQ(idx, 8);

    source.reset();
}

TEST_P(MetaSourceTest, RepeatingSource) {
    RepeatingSource<TextEntity::Batch> source(
        3, /* num_repeats */
        new CountingSource(2));
    TextEntity::Batch batch(128, /* batch_size*/
                            3 /* window_size */);

    size_t idx = 0;
    while (source.has_next()) {
        source.next(&batch);

        EXPECT_THAT(get_features(batch),
                    ElementsAreArray(std::vector<WordIdxType>(128 * 3, idx % 2)));

        EXPECT_THAT(get_labels(batch),
                    ElementsAreArray(std::vector<WordIdxType>(128, idx % 2)));

        EXPECT_THAT(get_weights(batch),
                    ElementsAreArray(std::vector<WordIdxType>(128, 1.0)));

        batch.clear();

        ++idx;
    }

    CHECK_EQ(idx, 6);

    source.reset();
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    CHECK(!FLAGS_test_data_dir.empty());
    return RUN_ALL_TESTS();
}
