#include <algorithm>
#include <deque>
#include <queue>

using namespace __gnu_cxx;
#include <indri/CompressedCollection.hpp>
#include <indri/DiskIndex.hpp>
#include <indri/QueryEnvironment.hpp>
#include <indri/Path.hpp>
#include <indri/Parameters.hpp>

#include <glog/logging.h>

#include "cuNVSM/data.h"

namespace indri {

api::Parameters LoadParameters(const std::string& repository_path) {
    CHECK(!repository_path.empty());

    const std::string parameter_path = indri::file::Path::combine(repository_path, "manifest");

    api::Parameters parameters;

    try {
        parameters.loadFile(parameter_path);
    } catch (const lemur::api::Exception& e) {
        LOG(FATAL) << "Unable to open Indri parameters: " << e.what();
    }

    return parameters;
}

index::DiskIndex* LoadIndex(const std::string& repository_path,
                            api::Parameters&& parameters) {
    api::Parameters container = parameters["indexes"];

    std::string index_path;

    if(container.exists("index")) {
        api::Parameters indexes = container["index"];

        if (indexes.size() != 1) {
            LOG(FATAL) << "Indri repository contain more than one index.";
        }

        index_path = indri::file::Path::combine(
            index_path, (std::string) indexes[static_cast<size_t>(0)]);
    } else {
        LOG(FATAL) << "Indri repository does not contain an index.";
    }

    index::DiskIndex* const index = new index::DiskIndex;

    try {
        index->open(indri::file::Path::combine(repository_path, "index"),
                    index_path);
    } catch (const lemur::api::Exception& e) {
        LOG(FATAL) << "Unable to open Indri index: " << e.what();
    }

    return index;
}

index::DiskIndex* LoadIndex(const std::string& repository_path) {
    return LoadIndex(repository_path, LoadParameters(repository_path));
}

uint64 GetDocumentCount(const std::string& repository_path) {
    std::unique_ptr<indri::index::DiskIndex> index(indri::LoadIndex(repository_path));

    const uint64 document_count = index->documentCount();

    index->close();

    return document_count;
}

api::QueryEnvironment* LoadQueryEnvironment(const std::string& repository_path) {
    indri::api::QueryEnvironment* query_env = new indri::api::QueryEnvironment;

    try {
        query_env->addIndex(repository_path);
    } catch (const lemur::api::Exception& e) {
        LOG(FATAL) << "Unable to open Indri query environment: " << e.what();
    }

    return query_env;
}

collection::CompressedCollection* LoadCollection(const std::string& repository_path) {
    indri::collection::CompressedCollection* collection =
        new indri::collection::CompressedCollection;

    const std::string collection_path =
        indri::file::Path::combine(repository_path, "collection");

    try {
        collection->open(collection_path);
    } catch (const lemur::api::Exception& e) {
        LOG(FATAL) << "Unable to open Indri collection: " << e.what();
    }

    return collection;
}

}  // namespace indri

namespace TextEntity {

InstanceGeneratorBase::InstanceGeneratorBase(IndriSource* const indri_source)
        : indri_source_(indri_source) {
    CHECK_NOTNULL(indri_source_);
};

void InstanceGeneratorBase::generate_terms(const IndriSource::DOCID_T int_doc_id,
                                           const indri::index::TermList& term_list,
                                           std::vector<WordIdxType>* const terms) const {
    CHECK(terms->empty());

    for (const IndriSource::TERMID_T term_id : term_list.terms()) {
        ssize_t token_id = -1;

        if (contains_key(indri_source_->term_id_mapping_, term_id)) {
            token_id = indri_source_->term_id_mapping_.at(term_id);
        } else if (indri_source_->include_oov_) {
            token_id = 0;
        } else {
            continue;
        }

        DCHECK_GE(token_id, 0);

        terms->push_back(token_id);
    }
}

class SequentialInstanceGenerator : public InstanceGeneratorBase {
 public:
  SequentialInstanceGenerator(IndriSource* const indri_source)
        : InstanceGeneratorBase(indri_source),
          current_document_(indri_source->document_id_mapping_.begin()),
          end_(indri_source->document_id_mapping_.end()) {
      reset();
  }

  virtual inline void generate(InstancesT* const instances) {
      const ObjectIdxType label = current_document_->first;
      const IndriSource::DOCID_T int_doc_id = current_document_->second;

      std::unique_ptr<const indri::index::TermList> term_list(
          this->indri_source_->index_->termList(current_document_->second));

      std::vector<WordIdxType> terms;

      generate_terms(int_doc_id, *term_list, &terms);

      const int32 object_length = this->indri_source_->document_lengths_[label];
      DCHECK_GE(object_length, this->indri_source_->window_size_);

      create_instances(terms, label,
                       exp(log(this->indri_source_->avg_document_length_) -
                           log(object_length)),
                       1 /* stride */,
                       instances);

      ++current_document_;
  }

  virtual bool has_next() const {
      return current_document_ != end_;
  }

  virtual void reset() {
      current_document_ = this->indri_source_->document_id_mapping_.begin();
  }

 private:
  template <typename Iterable>
  void create_instances(const Iterable& tokens,
                        const ObjectIdxType object_id,
                        const WeightType weight,
                        const size_t stride,
                        InstancesT* const instances) {
      DCHECK_GT(stride, 0);
      DCHECK_LE(stride, this->indri_source_->window_size_);

      std::deque<WordIdxType> buffer;

      for (const size_t token : tokens) {
          buffer.push_back(token);

          if (buffer.size() == this->indri_source_->window_size_) {
              instances->emplace_back(
                  std::vector<WordIdxType>(buffer.begin(), buffer.end()),
                  this->indri_source_->compute_term_weights(buffer),
                  object_id,
                  weight);

              // Stride forward.
              for (size_t i = 0; i < stride; ++i) {
                  buffer.pop_front();
              }
          }

          DCHECK_LE(buffer.size(), this->indri_source_->window_size_);
      }

      if (buffer.size() == this->indri_source_->window_size_) {
          instances->emplace_back(
              std::vector<WordIdxType>(buffer.begin(), buffer.end()),
              this->indri_source_->compute_term_weights(buffer),
              object_id,
              weight);
      }

      // TODO(cvangysel): maybe do something here with what is left in buffer.
  }

  IndriSource::DocumentIdMapping::iterator current_document_;
  const IndriSource::DocumentIdMapping::iterator end_;
};

class StochasticInstanceGenerator : public InstanceGeneratorBase {
 public:
  StochasticInstanceGenerator(const SamplingStrategy sampling_strategy,
                              const WeightingStrategy weighting_strategy,
                              IndriSource* const indri_source,
                              RNG* const rng)
        : InstanceGeneratorBase(indri_source),
          sampling_strategy_(sampling_strategy), weighting_strategy_(weighting_strategy),
          avg_document_length_(0.0),
          rng_(rng) {
      CHECK_NE(sampling_strategy, AUTOMATIC_SAMPLING);
      CHECK_NE(weighting_strategy, AUTOMATIC_WEIGHTING);

      CHECK_NOTNULL(this->indri_source_);

      size_t num_terms = 0;
      size_t num_document_too_short = 0;

      LOG(INFO) << "Loading documents into memory.";
      for (const auto& pair : indri_source->document_id_mapping_) {
          const ObjectIdxType label = pair.first;
          const IndriSource::DOCID_T int_doc_id = pair.second;

          std::unique_ptr<const indri::index::TermList> term_list(
              indri_source->index_->termList(int_doc_id));

          // Load term list.
          term_lists_.emplace(
              std::piecewise_construct,
              std::forward_as_tuple(label),
              std::forward_as_tuple());

          std::vector<WordIdxType>& terms = term_lists_.at(label);

          generate_terms(int_doc_id, *term_list, &terms);

          if (terms.size() < indri_source->window_size_) {
              LOG(WARNING) << "Document " << int_doc_id << " only has "
                           << terms.size() << " in-vocabulary tokens.";

              term_lists_.erase(label);

              ++num_document_too_short;

              continue;
          }

          num_terms += terms.size();
      }

      LOG(INFO) << "Unable to generate n-grams for " << num_document_too_short << " as they were too short.";

      *const_cast<float64*>(&avg_document_length_) =
          num_terms / static_cast<float64>(term_lists_.size());

      reset();
  }

  virtual inline void generate(InstancesT* const instances) {
      const size_t num = min(instance_order_.size(), 102400ul);

      std::vector<WordIdxType> buffer;
      buffer.resize(this->indri_source_->window_size_, 0);

      for (size_t i = 0; i < num; ++i) {
          const ObjectIdxType label = std::get<0>(instance_order_.front());
          const ObjectIdxType term_source_label = std::get<1>(instance_order_.front());
          const uint16 position = std::get<2>(instance_order_.front());

          const auto& terms = term_lists_.at(term_source_label);
          DCHECK_GE(terms.size(), this->indri_source_->window_size_);

          // Copy n-gram to buffer.
          std::copy(
              terms.begin() + position,
              terms.begin() + position + this->indri_source_->window_size_,
              buffer.begin());

          const int32 object_length = term_lists_.at(label).size();

          WeightType weight = 1.0;

          if (weighting_strategy_ == INV_DOC_FREQUENCY) {
              weight = exp(
                  log(avg_document_length_) -
                  log(object_length));
          } else if (weighting_strategy_ == UNIFORM) {
              // Do nothing.
          }

          instances->emplace_back(
              buffer,
              this->indri_source_->compute_term_weights(buffer),
              label,
              weight);

          instance_order_.pop_front();
      }
  }

  virtual bool has_next() const {
      return !instance_order_.empty();
  }

  virtual void reset() {
      if (!instance_order_.empty()) {
          LOG(WARNING) << "Resetting instance generator while there are still instances to consume.";

          instance_order_.clear();
      }

      // For NGRAM_FREQUENCY resampling strategy.
      const int32 num_samples = max(
          static_cast<int32>(ceil(
              avg_document_length_ -
              this->indri_source_->window_size_ + 1)),
          1l);

      if (sampling_strategy_ == NONE) {
          LOG(INFO) << "Generating instance pointers.";
      } else {
          LOG(INFO) << "Generating instance pointers "
                    << "(" << num_samples << " samples per document).";
      }

      std::vector<ObjectIdxType> object_identifiers;
      object_identifiers.resize(term_lists_.size());

      for (const auto& pair : term_lists_) {
          const ObjectIdxType label = pair.first;
          object_identifiers.push_back(label);
      }

      for (const auto& pair : term_lists_) {
          const ObjectIdxType label = pair.first;
          const IndriSource::DOCID_T int_doc_id = this->indri_source_->document_id_mapping_.at(pair.first);

          const int document_length = term_lists_.at(label).size();

          // The index document length is greater or equal to the document length
          // we record due to OoV. With the top-200k terms, approximately 50% of terms is OoV.
          CHECK_GE(this->indri_source_->document_lengths_.at(label), document_length);

          const int32 max_position = document_length - this->indri_source_->window_size_ + 1;

          if (sampling_strategy_ == NONE) {
              // TODO(cvangysel): figure out why we have this requirement?
              if (document_length >= (1 << 16)) {
                  LOG(WARNING) << "Skipping instance generation from object " << label 
                               << " as it exceeds 2^16 terms (" << document_length << ").";

                  continue;
              }

              CHECK_LT(document_length, 1 << 16);

              for (int32 position = 0;
                   position < max_position;
                   position += 1 /* stride */) {
                  instance_order_.emplace_back(label, label, position);
              }
          } else if (sampling_strategy_ == NGRAM_FREQUENCY) {
              std::uniform_int_distribution<int> term_position_distribution(0, max_position - 1);

              for (int32 i = 0; i < num_samples; ++i) {
                  instance_order_.emplace_back(label, label, term_position_distribution(*rng_));
              }
          } else {
              LOG(FATAL) << "Invalid sampling strategy: " << sampling_strategy_;
          }
      }

      LOG(INFO) << "Shuffling " << instance_order_.size() << " instance pointers.";
      std::shuffle(instance_order_.begin(), instance_order_.end(), *rng_);
  }

 private:
  const SamplingStrategy sampling_strategy_;
  const WeightingStrategy weighting_strategy_;

  const float64 avg_document_length_;

  std::map<ObjectIdxType, std::vector<WordIdxType>> term_lists_;
  std::deque<std::tuple<ObjectIdxType, ObjectIdxType, uint16>> instance_order_;

  RNG* const rng_;
};

IndriSource::IndriSource(const std::string& repository_path,
                         const size_t window_size,
                         RNG* const rng,
                         const size_t max_vocabulary_size,
                         const size_t min_document_frequency,
                         const size_t max_document_frequency,
                         const size_t documents_cutoff,
                         const bool include_oov,
                         const bool include_digits,
                         const std::vector<std::string>* const document_list,
                         const TermBlacklist* const term_blacklist,
                         const bool shuffle,
                         const SamplingStrategy sampling_strategy,
                         const WeightingStrategy weighting_strategy,
                         const TermWeightingStrategy term_weighting_strategy)
        : TextEntity::DataSource(0, /* vocabulary_size */
                                 0 /* corpus_size */),
          window_size_(window_size),
          include_oov_(include_oov),
          index_(indri::LoadIndex(repository_path)),
          query_env_(indri::LoadQueryEnvironment(repository_path)),
          collection_(indri::LoadCollection(repository_path)),
          num_terms_emitted_(0),
          total_num_terms_(0),
          avg_document_length_(0.0),
          instance_generator_(nullptr),
          term_weighting_strategy_(term_weighting_strategy) {
    initialize(max_vocabulary_size,
               min_document_frequency,
               max_document_frequency,
               include_digits,
               documents_cutoff,
               shuffle,
               sampling_strategy,
               weighting_strategy,
               document_list,
               term_blacklist,
               rng);
}

IndriSource::IndriSource(indri::index::DiskIndex* const index,
                         const size_t window_size,
                         RNG* const rng,
                         const size_t max_vocabulary_size,
                         const size_t min_document_frequency,
                         const size_t max_document_frequency,
                         const size_t documents_cutoff,
                         const bool include_oov,
                         const bool include_digits,
                         const bool shuffle,
                         const SamplingStrategy sampling_strategy,
                         const WeightingStrategy weighting_strategy,
                         const TermWeightingStrategy term_weighting_strategy)
        : TextEntity::DataSource(0 /* vocabulary_size */,
                                 0 /* corpus_size */),
          window_size_(window_size),
          include_oov_(include_oov),
          index_(index),
          query_env_(nullptr),
          collection_(nullptr),
          total_num_terms_(0),
          avg_document_length_(0.0),
          instance_generator_(nullptr),
          term_weighting_strategy_(term_weighting_strategy) {
    initialize(max_vocabulary_size,
               min_document_frequency,
               max_document_frequency,
               include_digits,
               documents_cutoff,
               shuffle,
               sampling_strategy,
               weighting_strategy,
               nullptr, /* document_list */
               nullptr, /* term_blacklist */
               rng);
}

IndriSource::~IndriSource() {
    index_->close();

    if (query_env_.get() != nullptr) {
        query_env_->close();
    }
}

void IndriSource::reset() {
    instance_generator_->reset();

    num_terms_emitted_ = 0;
}

void IndriSource::next(Batch* const batch) {
    CHECK(!term_id_mapping_.empty());
    CHECK_EQ(batch->window_size(), window_size_);

    TextEntity::DataSource::next(batch);

    InstancesT instances;

    while (!batch->full() && has_next()) {
        instance_generator_->generate(&instances);

        while (!instances.empty()) {
            push_instance(std::get<0>(instances.front()),
                          std::get<1>(instances.front()),
                          std::get<2>(instances.front()),
                          std::get<3>(instances.front()),
                          batch);

            instances.pop_front();
        }

        // TODO(cvangysel): fix this.
        // num_terms_emitted_ += terms.size();
    }
}

bool IndriSource::has_next() const {
    return TextEntity::DataSource::has_next() ||
        instance_generator_->has_next();
}

void IndriSource::extract_metadata(lse::Metadata* const metadata) const {
    // index term -> model term -> model object -> index object
    for (const auto& pair : term_id_mapping_) {
        lse::Metadata::TermInfo* const term = metadata->add_term();

        term->set_index_term_id(pair.first);
        term->set_model_term_id(pair.second);

        CHECK(contains_key(inv_term_id_to_term_freq_, pair.second));

        term->set_term_frequency(inv_term_id_to_term_freq_.at(pair.second));
    }

    metadata->set_total_terms(total_num_terms_);

    for (const auto& pair : document_id_mapping_) {
        lse::Metadata::ObjectInfo* const object = metadata->add_object();

        object->set_model_object_id(pair.first);
        object->set_index_object_id(pair.second);
    }
}

IdentifiersMapT* IndriSource::build_term_identifiers_map() const {
    CHECK_NOTNULL(collection_.get());

    std::unique_ptr<IdentifiersMapT> identifiers_map(new IdentifiersMapT);

    for (const auto& pair : term_id_mapping_) {
        const WordIdxType term_id = pair.second;

        const std::string term = index_->term(pair.first);

        insert_or_die(term, term_id,
                      identifiers_map.get());
    }

    return identifiers_map.release();
}

IdentifiersMapT* IndriSource::build_document_identifiers_map() const {
    CHECK_NOTNULL(collection_.get());

    std::unique_ptr<IdentifiersMapT> identifiers_map(new IdentifiersMapT);

    for (const auto& pair : document_id_mapping_) {
        const lemur::api::DOCID_T int_document_id = pair.second;

        const std::string ext_document_id =
            collection_->retrieveMetadatum(int_document_id, "docno");

        insert_or_die(ext_document_id, pair.first,
                      identifiers_map.get());
    }

    return identifiers_map.release();
}

size_t IndriSource::compute_term_frequency(const TERMID_T term_id) {
    CHECK_NOTNULL(index_.get());
    CHECK_GT(document_id_mapping_.size(), 0);

    size_t frequency = 0;

    std::unique_ptr<indri::index::DocListIterator> term_doc_list_it(
        index_->docListIterator(term_id));

    term_doc_list_it->startIteration();

    while (!term_doc_list_it->finished()) {
        if (!contains_key(document_id_mapping_,
                          term_doc_list_it->currentEntry()->document)) {
            // Ignore.
        } else {
            frequency += term_doc_list_it->currentEntry()->positions.size();
        }

        term_doc_list_it->nextEntry();
    }

    VLOG(2) << "Term " << term_doc_list_it->termData()->term << " "
            << "has frequency " << frequency << ".";

    return frequency;
}

void IndriSource::initialize(const size_t max_vocabulary_size,
                             const size_t min_document_frequency,
                             const size_t max_document_frequency,
                             const bool include_digits,
                             const size_t documents_cutoff,
                             const bool shuffle,
                             SamplingStrategy sampling_strategy,
                             WeightingStrategy weighting_strategy,
                             const std::vector<std::string>* const document_list,
                             const TermBlacklist* const term_blacklist,
                             RNG* const rng) {
    CHECK_NOTNULL(index_.get());

    CHECK(term_id_mapping_.empty());
    CHECK(inv_term_id_mapping_.empty());

    CHECK(inv_term_id_to_term_freq_.empty());

    CHECK(document_id_mapping_.empty());

    if (sampling_strategy == AUTOMATIC_SAMPLING) {
        sampling_strategy = shuffle ? NGRAM_FREQUENCY : NONE;
    }

    if (weighting_strategy == AUTOMATIC_WEIGHTING) {
        weighting_strategy = sampling_strategy == NONE ? INV_DOC_FREQUENCY : UNIFORM;
    }

    //
    // Initialize documents.
    //

    {
        LOG(INFO) << "Building document-id mapping for Indri.";

        const size_t document_list_size = (document_list == nullptr) ?
            index_->documentCount() : document_list->size();

        const size_t num_documents = min(
            min(documents_cutoff > 0 ? documents_cutoff : index_->documentCount(),
                index_->documentCount()),
            document_list_size);

        document_lengths_.resize(num_documents, 0L);

        size_t document_length_agg = 0;
        size_t model_doc_id = 0L;

        size_t discarded_documents = 0;

        if (document_list == nullptr) {
            DOCID_T index_doc_id = index_->documentBase();
            DOCID_T max_doc_id = index_->documentMaximum();

            while (document_id_mapping_.size() < num_documents &&
                   index_doc_id < max_doc_id) {
                const int document_length = index_->documentLength(index_doc_id);

                if (document_length >= window_size_) {
                    document_id_mapping_.insert(std::make_pair(model_doc_id, index_doc_id));

                    document_lengths_[model_doc_id] = document_length;
                    document_length_agg += document_length;

                    ++model_doc_id;
                } else {
                    ++discarded_documents;
                }

                ++index_doc_id;
            }
        } else {
            CHECK_NOTNULL(query_env_.get());

            std::vector<lemur::api::DOCID_T> int_doc_ids =
                query_env_->documentIDsFromMetadata("docno", *document_list);

            CHECK_EQ(int_doc_ids.size(), document_list->size());

            for (const lemur::api::DOCID_T index_doc_id : int_doc_ids) {
                if (document_id_mapping_.size() >= num_documents) {
                    break;
                }

                CHECK(!contains_key(document_id_mapping_, model_doc_id));

                const int document_length = index_->documentLength(index_doc_id);
                if (document_length >= window_size_) {
                    document_id_mapping_.insert(std::make_pair(model_doc_id, index_doc_id));

                    document_lengths_[model_doc_id] = document_length;
                    document_length_agg += document_length;

                    ++model_doc_id;
                } else {
                    ++discarded_documents;
                }
            }
        }

        LOG(INFO) << "Discarded " << discarded_documents << " documents which were too short.";

        CHECK_LE(document_id_mapping_.size(), num_documents);

        *const_cast<size_t*>(&corpus_size_) = document_id_mapping_.size();
        *const_cast<WeightType*>(&avg_document_length_) =
            document_length_agg / static_cast<WeightType>(document_id_mapping_.size());

        CHECK_GT(avg_document_length_, 0.0);
    }

    //
    // Initialize vocabulary.
    //

    {
        LOG(INFO) << "Building term-id mapping for Indri.";
        size_t num_terms = 0;

        const size_t corpus_unique_term = index_->uniqueTermCount() + 1;

        typedef std::pair<int32, size_t> TermInfo;

        std::priority_queue<TermInfo,
                            std::vector<TermInfo>,
                            std::greater<TermInfo>> pq;

        std::unique_ptr<indri::index::VocabularyIterator> vocabulary_it(
            index_->vocabularyIterator());

        vocabulary_it->startIteration();

        size_t num_terms_discarded_zero = 0;
        size_t num_terms_discarded_blacklist = 0;
        size_t num_terms_discarded_digits = 0;
        size_t num_terms_discarded_document_frequency_too_high = 0;
        size_t num_terms_discarded_document_frequency_too_low = 0;

        while (!vocabulary_it->finished()) {
            indri::index::DiskTermData* const term_data =
                vocabulary_it->currentEntry();

            const TERMID_T term_id = term_data->termID;

            if (term_id == 0) {
                // Ignore.

                ++num_terms_discarded_zero;
            } else if (!include_digits && is_number(term_data->termData->term)) {
                // Ignore.

                ++num_terms_discarded_digits;
            } else if (min_document_frequency > 0 && term_data->termData->corpus.documentCount < min_document_frequency) {
                // Ignore.

                ++num_terms_discarded_document_frequency_too_low;
            } else if (max_document_frequency > 0 && term_data->termData->corpus.documentCount > max_document_frequency) {
                // Ignore.

                ++num_terms_discarded_document_frequency_too_high;
            } else {
                if (term_blacklist != nullptr && contains_key(*term_blacklist, term_data->termData->term)) {
                    // Ignore.
                    ++ num_terms_discarded_blacklist;

                    continue;
                }

                const size_t frequency = term_data->termData->corpus.totalCount;
                CHECK_GT(frequency, 0);

                if (max_vocabulary_size && (corpus_unique_term > max_vocabulary_size)) {
                    if ((pq.size() >= max_vocabulary_size) &&
                        (std::get<0>(pq.top()) < frequency)) {
                        pq.pop();
                    }

                    if (pq.size() < max_vocabulary_size) {
                        pq.push(std::make_pair(frequency, term_id));
                    }

                    DCHECK_LE(pq.size(), max_vocabulary_size);
                } else {
                    pq.push(std::make_pair(frequency, term_id));
                }
            }

            vocabulary_it->nextEntry();
        }

        if (max_vocabulary_size) {
            CHECK_LE(pq.size(), max_vocabulary_size);
        }

        if (include_oov_) {
            CHECK(term_id_mapping_.empty());

            term_id_mapping_.insert(std::make_pair(0 /* indri_term_id */, 0 /* our_term_id */));

            inv_term_id_mapping_.insert(std::make_pair(0 /* our_term_id */, 0 /* indri_term_id */));

            // Store collection frequency.
            inv_term_id_to_term_freq_.insert(std::make_pair(0 /* our_term_id */, 1 /* frequency */));
        }

        while (!pq.empty()) {
            const size_t indri_term_id = std::get<1>(pq.top());
            const size_t our_term_id = term_id_mapping_.size();

            size_t frequency = 0;
            if (corpus_size() == index_->documentCount()) {
                // Happy days scenario.
                frequency = std::get<0>(pq.top());
            } else {
                frequency = compute_term_frequency(indri_term_id);
            }

            pq.pop();

            if (frequency == 0) {
                continue;
            }

            num_terms += frequency;

            term_id_mapping_.insert(std::make_pair(
                indri_term_id, our_term_id));

            inv_term_id_mapping_.insert(std::make_pair(
                our_term_id, indri_term_id));

            // Store collection frequency.
            inv_term_id_to_term_freq_.insert(std::make_pair(
                our_term_id, frequency));
        }

        LOG(INFO) << "Vocabulary filtering discarded "
                  << num_terms_discarded_zero << " meta-terms, "
                  << num_terms_discarded_blacklist << " blacklisted terms, "
                  << num_terms_discarded_digits << " terms that contained a digit, "
                  << num_terms_discarded_document_frequency_too_high << " terms that had too high document frequency, "
                  << num_terms_discarded_document_frequency_too_low << " terms that had too low document frequency.";

        *const_cast<size_t*>(&vocabulary_size_) = term_id_mapping_.size();

        CHECK_EQ(term_id_mapping_.empty(), inv_term_id_mapping_.empty());

        CHECK_GT(num_terms, 0);
        *const_cast<size_t*>(&total_num_terms_) = num_terms;
    }

    const float64 log_corpus_vocabulary_ratio =
        log10(static_cast<float64>(total_num_terms_)) -
        log10(static_cast<float64>(vocabulary_size_));

    LOG(INFO) << "Index contains " << vocabulary_size_ << " unique terms and "
              << total_num_terms_ << " term occurrences "
              << "(log-ratio=" << log_corpus_vocabulary_ratio << ").";

    if (!shuffle) {
        CHECK_EQ(sampling_strategy, NONE);

        instance_generator_.reset(new SequentialInstanceGenerator(this));
    } else {
        instance_generator_.reset(new StochasticInstanceGenerator(
            sampling_strategy, weighting_strategy, this, rng));
    }
}

IndriSource::TERMID_T IndriSource::term_id(const std::string& term) const {
    CHECK(!term_id_mapping_.empty());

    // const cast to deal with crappy Indri class definitions.
    const int32 indri_term_id = const_cast<IndriSource*>(this)->index_->term(term);

    if (contains_key(term_id_mapping_, indri_term_id)) {
        return term_id_mapping_.at(indri_term_id);
    } else {
        return -1;
    }
}

std::string IndriSource::term(TERMID_T term_id) const {
    CHECK(!inv_term_id_mapping_.empty());

    CHECK(contains_key(inv_term_id_mapping_, term_id));
    return const_cast<IndriSource*>(this)->index_->term(inv_term_id_mapping_.at(term_id));
}

}  // namespace TextEntity

namespace EntityEntity {

typedef ::TextEntity::IndriSource::TERMID_T TERMID_T;

}  // namespace EntityEntity
