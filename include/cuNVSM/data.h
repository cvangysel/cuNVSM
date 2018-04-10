#ifndef CUNVSM_DATA_H
#define CUNVSM_DATA_H

#include <algorithm>
#include <deque>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <sstream>
#include <thread>

#include <boost/lockfree/queue.hpp>

#include <glog/logging.h>
#include <gtest/gtest_prod.h>

#include "cuNVSM/base.h"
#include "nvsm.pb.h"

typedef int32 WordIdxType;
typedef int32 ObjectIdxType;
typedef FLOATING_POINT_TYPE WeightType;

typedef std::map<std::string, int32> IdentifiersMapT;

// Forward declarations.
template <typename BatchT>
class DataSource;

// For testing.
template <typename ParentClass>
class DataSourceTestHelper;

namespace indri {
namespace index {
class DiskIndex;
class TermList;
}  // namespace index
namespace api {
class QueryEnvironment;
class Parameters;
}  // namespace api
namespace collection {
class CompressedCollection;
}  // namespace collection
}  // namespace indri

class BatchInterface {
 public:
  virtual ~BatchInterface() {}

  virtual void clear() = 0;

  virtual bool full() const = 0;
  virtual bool empty() const = 0;

  virtual void swap(BatchInterface* const other_batch) = 0;

  virtual inline size_t num_instances() const = 0;

  virtual inline size_t maximum_size() const = 0;
};

class DataSourceInterface {
 public:
  virtual ~DataSourceInterface() {}

  virtual void reset() = 0;

  virtual bool has_next() const = 0;

  virtual float32 progress() const = 0;

  virtual void extract_metadata(lse::Metadata* const metadata) const = 0;
};

template <typename BatchT>
class DataSource : public DataSourceInterface {
 public:
  typedef BatchT BatchType;

  virtual void next(BatchT* const batch) = 0;
};

namespace indri {

api::Parameters LoadParameters(const std::string& repository_path);

index::DiskIndex* LoadIndex(const std::string& repository_path,
                            api::Parameters&& parameters);
index::DiskIndex* LoadIndex(const std::string& repository_path);

uint64 GetDocumentCount(const std::string& repository_path);

api::QueryEnvironment* LoadQueryEnvironment(const std::string& repository_path);

collection::CompressedCollection* LoadCollection(const std::string& repository_path);

}  // namespace indri

namespace TextEntity {

typedef std::tuple<std::vector<WordIdxType>,
                   std::vector<WeightType>,
                   ObjectIdxType,
                   WeightType> InstanceT;
typedef std::deque<InstanceT> InstancesT;

// Forward declarations.
class DataSource;
class Objective;

class Batch : public BatchInterface {
 public:
  Batch(const size_t batch_size, const size_t window_size);
  explicit Batch(const lse::TrainConfig& train_config);

  // Forward constructor.
  Batch(const std::tuple<size_t, size_t>& args)
      : Batch(std::get<0>(args), std::get<1>(args)) {}

  // Move constructor.
  Batch(Batch&& other);

  virtual ~Batch();

  virtual void clear() override;

  virtual bool full() const override;
  virtual bool empty() const override;

  virtual void swap(BatchInterface* const other) override;

  virtual inline size_t num_instances() const override {
      return num_instances_;
  }

  virtual inline size_t maximum_size() const override {
      return batch_size_;
  }

  inline size_t window_size() const {
      return window_size_;
  }

 private:
  const size_t batch_size_;
  const size_t window_size_;

  typedef WordIdxType* FeaturesType;
  typedef ObjectIdxType* LabelsType;
  typedef WeightType* WeightsType;

  FeaturesType features_;
  WeightsType feature_weights_;

  LabelsType labels_;

  WeightsType weights_;

  size_t num_instances_;

  friend class TextEntity::DataSource;
  friend class TextEntity::Objective;

  friend std::ostream& operator<<(std::ostream& os, const Batch& batch);

  // For testing purposes.
  template <typename ParentClass>
  friend class ::DataSourceTestHelper;
  FRIEND_TEST(IndriSourceTest, IndriSource);
  FRIEND_TEST(IndriSourceTest, Brown);

 private:
  DISALLOW_COPY_AND_ASSIGN(Batch);
};

std::ostream& operator<<(std::ostream& os, const Batch& batch);

class DataSource : public ::DataSource<Batch> {
 public:
  DataSource(
      const size_t vocabulary_size,
      const size_t corpus_size)
      : vocabulary_size_(vocabulary_size),
        corpus_size_(corpus_size) {}

  virtual ~DataSource() {
      // CHECK(overflow_buffer_.empty());
  }

  virtual void next(Batch* const batch) {
      CHECK(batch->empty());

      while (!batch->full() && !overflow_buffer_.empty()) {
          push_instance(std::get<0>(*overflow_buffer_.begin()),
                        std::get<1>(*overflow_buffer_.begin()),
                        std::get<2>(*overflow_buffer_.begin()),
                        std::get<3>(*overflow_buffer_.begin()),
                        batch);

          overflow_buffer_.pop_front();
      }
  };

  virtual bool has_next() const {
      return !overflow_buffer_.empty();
  }

  inline size_t vocabulary_size() const {
      DCHECK_GT(vocabulary_size_, 0);

      return vocabulary_size_;
  }

  inline size_t corpus_size() const {
      DCHECK_GT(corpus_size_, 0);

      return corpus_size_;
  }

  virtual float32 progress() const {
      return NAN;
  }

  virtual void extract_metadata(lse::Metadata* const metadata) const {}

 protected:
  void push_instance(const std::vector<WordIdxType>& buffer,
                     const std::vector<FLOATING_POINT_TYPE> feature_weights,
                     const ObjectIdxType object_id,
                     const WeightType weight,
                     Batch* const batch);

  template <typename Iterable>
  void create_instances(const Iterable& tokens,
                        const ObjectIdxType object_id,
                        const WeightType weight,
                        const size_t stride,
                        Batch* const batch) {
      DCHECK_GT(batch->window_size(), 0);
      DCHECK_GT(stride, 0);

      DCHECK_LE(stride, batch->window_size());

      std::deque<WordIdxType> buffer;

      for (const size_t token : tokens) {
          buffer.push_back(token);

          if (buffer.size() == batch->window_size()) {
              push_instance(std::vector<WordIdxType>(buffer.begin(), buffer.end()),
                            std::vector<WeightType>(),
                            object_id, weight, batch);

              // Stride forward.
              for (size_t i = 0; i < stride; ++i) {
                  buffer.pop_front();
              }
          }

          CHECK_LE(buffer.size(), batch->window_size());
      }

      if (buffer.size() == batch->window_size()) {
          push_instance(std::vector<WordIdxType>(buffer.begin(), buffer.end()),
                        std::vector<WeightType>(),
                        object_id, weight, batch);
      }

      // TODO(cvangysel): maybe do something here with what is left in buffer.
  }

  const size_t vocabulary_size_;
  const size_t corpus_size_;

  InstancesT overflow_buffer_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DataSource);
};

typedef std::map<std::string, WordIdxType> VocabularyT;
typedef std::vector<std::pair<ObjectIdxType, std::string>> CorpusT;

inline VocabularyT construct_vocabulary(const std::vector<std::string>& words) {
    VocabularyT vocabulary;

    vocabulary["<UNK>"] = 0L;

    for (const std::string& word : words) {
        if (!contains_key(vocabulary, word)) {
            vocabulary.insert({word, vocabulary.size()});
        }
    }

    return vocabulary;
}

class InMemoryDocumentSource : public DataSource {
 public:
  InMemoryDocumentSource(const VocabularyT& vocabulary,
                         const CorpusT& documents,
                         const bool pad_batch = false)
      : DataSource(vocabulary.size(), documents.size()),
        vocabulary_(vocabulary),
        documents_(documents),
        pad_batch_(pad_batch) {
      reset();
  }

  virtual void reset() override {
    num_batches_emitted_ = 0;
  }

  virtual void next(Batch* const batch) override {
      DataSource::next(batch);

      // Do at least one loop, and then continue if pad_batch_ is true.
      while (batch->num_instances() == 0 ||
             (pad_batch_ &&
              batch->num_instances() < batch->maximum_size())) {
          for (auto& document : documents_) {
              const ObjectIdxType doc_id = std::get<0>(document);
              std::string contents = std::get<1>(document);

              std::vector<WordIdxType> tokens;

              for (const std::string& word : split(contents)) {
                  if (!contains_key(vocabulary_, word)) {
                      continue;
                  }

                  const WordIdxType token = vocabulary_.at(word);

                  tokens.push_back(token);
              }

              const WeightType weight = exp(-log(tokens.size()));

              create_instances(tokens, doc_id, weight,
                               1 /* stride */,
                               batch);
          }
      }

      ++num_batches_emitted_;
  }

  virtual bool has_next() const override {
      return DataSource::has_next() || num_batches_emitted_ < 2;
  }

 private:
  size_t num_batches_emitted_;

  const VocabularyT vocabulary_;
  const CorpusT documents_;

  const bool pad_batch_;

  DISALLOW_COPY_AND_ASSIGN(InMemoryDocumentSource);
};

// Forward declaration.
class InstanceGeneratorBase;

enum SamplingStrategy {
    AUTOMATIC_SAMPLING, NONE, NGRAM_FREQUENCY
};

enum WeightingStrategy {
    AUTOMATIC_WEIGHTING, UNIFORM, INV_DOC_FREQUENCY
};

enum TermWeightingStrategy {
    UNIFORM_TERM_WEIGHTING, SELF_INFORMATION_TERM_WEIGHTING
};

class IndriSource : public DataSource {
 public:
  typedef int32 /* lemur::api:: */ TERMID_T;
  typedef int32 /* lemur::api:: */ DOCID_T;

  typedef std::map<TERMID_T, size_t> TermIdMapping;
  typedef std::map<size_t, DOCID_T> DocumentIdMapping;

  typedef std::set<std::string> TermBlacklist;

  IndriSource(const std::string& repository_path,
              const size_t window_size,
              RNG* const rng,
              const size_t max_vocabulary_size = 0,
              const size_t min_document_frequency = 0,
              const size_t max_document_frequency = 0,
              const size_t documents_cutoff = 0,
              const bool include_oov = false,
              const bool include_digits = false,
              const std::vector<std::string>* const document_list = nullptr,
              const TermBlacklist* const term_blacklist = nullptr,
              const bool shuffle = false,
              const SamplingStrategy sampling_strategy = AUTOMATIC_SAMPLING,
              const WeightingStrategy weighting_strategy = AUTOMATIC_WEIGHTING,
              const TermWeightingStrategy term_weighting_strategy = UNIFORM_TERM_WEIGHTING);

  // Takes ownership.
  IndriSource(indri::index::DiskIndex* const index,
              const size_t window_size,
              RNG* const rng,
              const size_t max_vocabulary_size = 0,
              const size_t min_document_frequency = 0,
              const size_t max_document_frequency = 0,
              const size_t documents_cutoff = 0,
              const bool include_oov = false,
              const bool include_digits = false,
              const bool shuffle = false,
              const SamplingStrategy sampling_strategy = AUTOMATIC_SAMPLING,
              const WeightingStrategy weighting_strategy = AUTOMATIC_WEIGHTING,
              const TermWeightingStrategy term_weighting_strategy = UNIFORM_TERM_WEIGHTING);

  virtual ~IndriSource();

  virtual void reset() override;

  virtual void next(Batch* const) override;
  virtual bool has_next() const override;

  virtual float32 progress() const override {
      return num_terms_emitted_ / static_cast<float64>(total_num_terms_);
  }

  virtual void extract_metadata(lse::Metadata* const metadata) const override;

  TERMID_T term_id(const std::string& term) const;
  std::string term(/* lemur::api::TERMID_T */ int32 term_id) const;

  const TermIdMapping& term_id_mapping() const {
      return term_id_mapping_;
  }

  const DocumentIdMapping& document_id_mapping() const {
      return document_id_mapping_;
  }

  IdentifiersMapT* build_term_identifiers_map() const;
  IdentifiersMapT* build_document_identifiers_map() const;

 protected:
  size_t compute_term_frequency(const TERMID_T term_id);

  void initialize(const size_t max_vocabulary_size,
                  const size_t min_document_frequency,
                  const size_t max_document_frequency,
                  const bool include_digits,
                  const size_t documents_cutoff,
                  const bool shuffle,
                  SamplingStrategy sampling_strategy,
                  WeightingStrategy weighting_strategy,
                  const std::vector<std::string>* const document_list,
                  const TermBlacklist* const term_blacklist,
                  RNG* const rng);

  template <typename Iterable>
  std::vector<WeightType> compute_term_weights(const Iterable& iterable) {
      if (term_weighting_strategy_ == UNIFORM_TERM_WEIGHTING) {
          return std::vector<WeightType>();
      } else if (term_weighting_strategy_ == SELF_INFORMATION_TERM_WEIGHTING) {
          // TODO(cvangysel): the code below can be optimized further.
          std::vector<WeightType> weights;

          for (const size_t term_id : iterable) {
              DCHECK(contains_key(inv_term_id_to_term_freq_, term_id));

              const WeightType self_information = - log(
                  static_cast<WeightType>(inv_term_id_to_term_freq_.at(term_id)) /
                  total_num_terms_);

              DCHECK(std::isfinite(self_information));

              weights.push_back(self_information);
          }

          return weights;
      } else {
          LOG(FATAL) << "Unknown term weighting strategy";
      }
  }

 private:
  std::unique_ptr<indri::index::DiskIndex> index_;
  std::unique_ptr<indri::api::QueryEnvironment> query_env_;
  std::unique_ptr<indri::collection::CompressedCollection> collection_;  // Required for build_identifiers_map().

  const size_t window_size_;

  const bool include_oov_;

  size_t num_terms_emitted_;

  const size_t total_num_terms_;
  const WeightType avg_document_length_;

  TermIdMapping term_id_mapping_;
  std::map<size_t, TERMID_T> inv_term_id_mapping_;

  std::map<size_t, int32> inv_term_id_to_term_freq_;

  std::vector<int> document_lengths_;

  DocumentIdMapping document_id_mapping_;

  std::unique_ptr<InstanceGeneratorBase> instance_generator_;

  const TermWeightingStrategy term_weighting_strategy_;

  friend class InstanceGeneratorBase;
  friend class SequentialInstanceGenerator;
  friend class StochasticInstanceGenerator;

  // For testing.
  friend class IndriSourceTest;
  FRIEND_TEST(IndriSourceTest, StochasticIndriSource_SelfInformation);

  DISALLOW_COPY_AND_ASSIGN(IndriSource);
};

class InstanceGeneratorBase {
 public:
  explicit InstanceGeneratorBase(IndriSource* const indri_source);
  virtual ~InstanceGeneratorBase() {};

  virtual void generate(InstancesT* const batch) = 0;

  virtual bool has_next() const = 0;

  virtual void reset() = 0;

 protected:
  void generate_terms(const IndriSource::DOCID_T int_doc_id,
                      const indri::index::TermList& term_list,
                      std::vector<WordIdxType>* const terms) const;

  IndriSource* indri_source_;

  DISALLOW_COPY_AND_ASSIGN(InstanceGeneratorBase);
};

}  // namespace TextEntity

namespace RepresentationSimilarity {

typedef std::tuple<ObjectIdxType, ObjectIdxType, WeightType> InstanceT;
typedef std::deque<InstanceT> InstancesT;

// Forward declarations.
class DataSource;
class Objective;

class Batch : public BatchInterface {
 public:
  explicit Batch(const size_t batch_size);
  explicit Batch(const lse::TrainConfig& train_config);

  // Forward constructor.
  Batch(const std::tuple<size_t>& args) : Batch(std::get<0>(args)) {};

  // Move constructor.
  Batch(Batch&& other);

  // For interfacing with AsyncSource; window_size is ignored.
  Batch(const size_t batch_size, const size_t window_size)
      : Batch(batch_size) {}

  virtual ~Batch();

  virtual void clear() override;

  virtual bool full() const override;
  virtual bool empty() const override;

  virtual void swap(BatchInterface* const other) override;

  virtual inline size_t num_instances() const override {
      return num_instances_;
  }

  virtual inline size_t maximum_size() const override {
      return batch_size_;
  }

 private:
  const size_t batch_size_;

  typedef ObjectIdxType* FeaturesType;
  typedef WeightType* WeightsType;

  FeaturesType features_;
  WeightsType weights_;

  size_t num_instances_;

  friend class RepresentationSimilarity::DataSource;
  friend class RepresentationSimilarity::Objective;

  friend std::ostream& operator<<(std::ostream& os, const Batch& batch);

  // For testing purposes.
  template <typename ParentClass>
  friend class ::DataSourceTestHelper;
  FRIEND_TEST(RepresentationSimilarity, DataSource);

  DISALLOW_COPY_AND_ASSIGN(Batch);
};

std::ostream& operator<<(std::ostream& os, const Batch& batch);

std::vector<InstanceT>* LoadSimilarities(
        const std::string& path,
        const IdentifiersMapT& identifiers_map);

std::vector<InstanceT>* LoadSimilarities(
        std::istream& file,
        const IdentifiersMapT& identifiers_map);

class DataSource : public ::DataSource<Batch> {
 public:
  DataSource(const std::string& path,
             const IdentifiersMapT& identifiers_map,
             RNG* const rng);

  // Takes ownership.
  DataSource(const std::vector<InstanceT>* const data,
             RNG* const rng);

  virtual ~DataSource() {}

  virtual void reset();

  virtual void next(Batch* const batch);

  virtual bool has_next() const;

  virtual float32 progress() const;

  virtual void extract_metadata(lse::Metadata* const metadata) const {}

 private:
  std::unique_ptr<const std::vector<InstanceT>> data_;
  RNG* const rng_;

  std::deque<size_t> instance_order_;
};

}  // namespace RepresentationSimilarity

namespace EntityEntity {

using RepresentationSimilarity::Batch;

}  // namespace EntityEntity

template <typename BatchT>
class AsyncSource : public DataSource<BatchT> {
 public:
  typedef BatchT BatchType;

  // Takes ownership.
  AsyncSource(const size_t num_concurrent_batches,
              const size_t batch_size,
              const size_t window_size,
              DataSource<BatchType>* const source);
  virtual ~AsyncSource();

  virtual void reset() override;

  virtual void next(BatchType* const) override;
  virtual bool has_next() const override;

  virtual float32 progress() const override;

  virtual void extract_metadata(lse::Metadata* const metadata) const override;

 protected:
  bool batches_ready() const;
  bool worker_running() const;

  void stop_worker();
  void start_worker();

 private:
  const size_t num_concurrent_batches_;

  std::unique_ptr<DataSource<BatchType>> source_;

  // For memory management.
  std::vector<std::unique_ptr<BatchType>> buffers_;

  typedef boost::lockfree::queue<BatchType*> result_queue;

  result_queue empty_batches_;
  result_queue full_batches_;

  std::unique_ptr<std::thread> thread_;

  std::atomic_bool is_running_;
  std::future<void> finished_;

  DISALLOW_COPY_AND_ASSIGN(AsyncSource);
};

template <typename ... BatchT>
class MultiSource : public DataSource<std::tuple<BatchT ...>> {
 public:
  typedef std::tuple<BatchT ...> BatchType;
  constexpr static size_t num_sources = std::tuple_size<std::tuple<BatchT ...>>::value;

  // Takes ownership.
  MultiSource(const std::tuple<DataSource<BatchT>* ...>& sources);
  virtual ~MultiSource();

  virtual void reset() override;

  virtual void next(BatchType* const) override;
  virtual bool has_next() const override;

  virtual float32 progress() const override;

  virtual void extract_metadata(lse::Metadata* const metadata) const override;

 private:
  std::tuple<std::unique_ptr<DataSource<BatchT>> ...> sources_;

  DISALLOW_COPY_AND_ASSIGN(MultiSource);
};

template <typename BatchT>
class RepeatingSource : public DataSource<BatchT> {
 public:
  typedef BatchT BatchType;

  // Takes ownership.
  RepeatingSource(const size_t num_repeats,
                  DataSource<BatchT>* source);

  virtual void reset() override;

  virtual void next(BatchType* const) override;
  virtual bool has_next() const override;

  virtual float32 progress() const override;

  virtual void extract_metadata(lse::Metadata* const metadata) const override;

 private:
  std::unique_ptr<DataSource<BatchT>> source_;
  const size_t num_repeats_;

  size_t current_iteration_;

  DISALLOW_COPY_AND_ASSIGN(RepeatingSource);
};

#endif /* CUNVSM_DATA_H */