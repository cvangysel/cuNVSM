#include <chrono>
#include <execinfo.h>
#include <typeinfo>
#include <iomanip>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cuda_profiler_api.h>

#include "cuNVSM/model.h"
#include "cuNVSM/hdf5.h"
#include "cuNVSM/gradient_check.h"

DEFINE_uint64(num_epochs, 100000, "Number of training iterations.");

DEFINE_uint64(document_cutoff, 0,  "Number of documents per epoch (default: all).");

DEFINE_string(document_list, "",  "Path to document list (default: all).");
DEFINE_string(term_blacklist, "",  "Path to term blacklist (default: none).");

DEFINE_uint64(word_repr_size, 4, "Dimensionality of word representations.");
DEFINE_uint64(entity_repr_size, 4, "Dimensionality of entity representations.");

DEFINE_uint64(batch_size, 1024, "Size of training batches.");
DEFINE_uint64(window_size, 8, "Size of training word windows.");

DEFINE_uint64(num_random_entities, 1, "Number of random negative examples sampled for each positive example.");

DEFINE_uint64(seed, 0, "Pseudo-random number generator seed.");

DEFINE_double(regularization_lambda, 0.01, "Regularization lambda.");
DEFINE_double(learning_rate, 0.0, "Learning rate.");

DEFINE_string(update_method, "", "Update method (sgd, adagrad, sparse_adam, dense_adam or full_adam).");

DEFINE_string(weighting, "auto", "Instance weighting strategy (auto, uniform or inv_doc_frequency).");

DEFINE_string(feature_weighting, "uniform", "Feature weighting strategy (uniform or self_information).");

DEFINE_bool(bias_negative_samples, false, "Introduces a bias towards negative samples. "
                                          "This is considered a bug in the CIKM model.");

DEFINE_string(nonlinearity, "", "Nonlinearity (tanh or hard_tanh).");

DEFINE_bool(l2_phrase_normalization, false, "Enables l2 normalization of phrase representations.");
DEFINE_bool(l2_entity_normalization, false, "Enables l2 normalization of entity representations.");

DEFINE_bool(batch_normalization, false, "Enables batch normalization.");

DEFINE_uint64(max_vocabulary_size, 60000, "Maximum vocabulary size.");

DEFINE_uint64(min_document_frequency, 2, "Minimum document frequency of term in order to be retained by vocabulary filtering.");

DEFINE_double(max_document_frequency, 0.5, "Maximum document frequency of term in order to be retained by vocabulary filtering. "
                                           "If smaller than 1.0, then max_document_frequency is interpreted as relative to the index size; "
                                           "otherwise, it is considered an absolute threshold.");

DEFINE_bool(include_oov, false, "Whether to include a special-purpose OoV token for term positions with a filtered dictionary term.");

DEFINE_bool(compute_initial_cost, false, "Compute the cost before any learning is performed.");

DEFINE_bool(check_gradients, false, "Enable gradient checking. "
                                    "CAUTION: this will lead to insanely slow learning.");

DEFINE_bool(no_shuffle, false, "Do not shuffle the training set.");

DEFINE_bool(dump_initial_model, false, "Dump the model after random initialization, but before training.");
DEFINE_int64(dump_every, 0, "Number of batches that should be processed "
                            "before the model is dumped during a single epoch. "
                            "The model is always dumped at the end of every epoch.");

DEFINE_double(entity_similarity_weight, 0.0, "Mixture weight of the entity-entity objective.");
DEFINE_double(term_similarity_weight, 0.0, "Mixture weight of the term-term objective.");

DEFINE_string(output, "", "Path to output model.");

template <typename BatchT>
class BatchHandler {
 public:
  static BatchT* create(const lse::TrainConfig& train_config) {
      return new BatchT(train_config);
  }

  static size_t num_instances(const BatchT& batch) {
      return batch.num_instances();
  }

  static void clear(BatchT& batch) {
      return batch.clear();
  }
};

template <typename FirstT, typename SecondT>
class BatchHandler<std::tuple<FirstT, SecondT>> {
 public:
  static std::tuple<FirstT, SecondT>* create(const lse::TrainConfig& train_config) {
      return new std::tuple<FirstT, SecondT>(
          FirstT(train_config), SecondT(train_config));
  }

  static size_t num_instances(const std::tuple<FirstT, SecondT>& batch) {
      return min(std::get<0>(batch).num_instances(),
                 std::get<1>(batch).num_instances());
  }

  static void clear(std::tuple<FirstT, SecondT>& batch) {
      std::get<0>(batch).clear();
      std::get<1>(batch).clear();
  }
};

void exception_handler() {
    void *trace_elems[20];
    int trace_elem_count(backtrace(trace_elems, 20));
    char **stack_syms(backtrace_symbols(trace_elems, trace_elem_count));
    for (int i = 0; i < trace_elem_count; ++i) {
        LOG(ERROR) << stack_syms[i];
    }
    free(stack_syms);

    try { throw; }
    catch (const thrust::system::system_error& e) {
        LOG(ERROR) << "Thurst system_error: " << e.what();
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Exception: " << typeid(e).name() << ": " << e.what();
    }
    catch (...) {
        LOG(ERROR) << "Unknown exception";
    }

    exit(1);
}

const std::map<std::string, TextEntity::WeightingStrategy> WEIGHTING_STRATEGIES {
    {"auto", TextEntity::AUTOMATIC_WEIGHTING},
    {"uniform", TextEntity::UNIFORM},
    {"inv_doc_frequency", TextEntity::INV_DOC_FREQUENCY},
};

const std::map<std::string, TextEntity::TermWeightingStrategy> FEATURE_WEIGHTING_STRATEGIES {
    {"uniform", TextEntity::UNIFORM_TERM_WEIGHTING},
    {"self_information", TextEntity::SELF_INFORMATION_TERM_WEIGHTING},
};

template <typename T>
T* read_strings(const std::string& path) {
    std::ifstream file(path);

    T* const strings = new T;

    std::string str;
    while (std::getline(file, str)) {
        if (!str.empty()) {
            insert_or_die(str, strings);
        }
    }

    return strings;
}

std::vector<std::string>* construct_document_list() {
    if (!FLAGS_document_list.empty()) {
        LOG(INFO) << "Reading document list from " << FLAGS_document_list << ".";

        return read_strings<std::vector<std::string>>(FLAGS_document_list);
    } else {
        return nullptr;
    }
}

TextEntity::IndriSource::TermBlacklist* construct_term_blacklist() {
    if (!FLAGS_term_blacklist.empty()) {
        LOG(INFO) << "Reading term blacklist from " << FLAGS_term_blacklist << ".";

        return read_strings<TextEntity::IndriSource::TermBlacklist>(FLAGS_term_blacklist);
    } else {
        return nullptr;
    }
}

TextEntity::IndriSource* construct_indri_source(
        const lse::DataConfig& data_config,
        const lse::TrainConfig& train_config,
        RNG* const rng) {
    std::unique_ptr<std::vector<std::string>> document_list(
        construct_document_list());

    std::unique_ptr<TextEntity::IndriSource::TermBlacklist> term_blacklist(
        construct_term_blacklist());

    return new TextEntity::IndriSource(
        data_config.repository_path(),
        train_config.window_size(),
        rng,
        data_config.max_vocabulary_size(),
        data_config.min_document_frequency(),
        data_config.max_document_frequency(),
        FLAGS_document_cutoff,
        data_config.include_oov(),
        false, /* include_digits */
        document_list.get(),
        term_blacklist.get(),
        !train_config.no_shuffle(), /* shuffle */
        TextEntity::AUTOMATIC_SAMPLING, /* sampling_method */
        WEIGHTING_STRATEGIES.at(FLAGS_weighting), /* weighting_strategy */
        FEATURE_WEIGHTING_STRATEGIES.at(FLAGS_feature_weighting) /* feature_weighting_strategy */);
}

template <typename BatchT>
DataSource<BatchT>* wrap_source_async(const lse::TrainConfig& train_config,
                                      DataSource<BatchT>* const source) {
    return new AsyncSource<BatchT>(
        10, /* num_concurrent_batches */
        train_config.batch_size(),
        train_config.window_size(),
        source);
}

template <typename ObjectiveT>
DataSource<typename ObjectiveT::BatchType>* construct_data_source(
        const lse::DataConfig& data_config,
        const lse::TrainConfig& train_config,
        RNG* const rng,
        const IdentifiersMapT* const identifiers_map = nullptr);

template <>
DataSource<typename TextEntity::Objective::BatchType>*
construct_data_source<TextEntity::Objective>(
        const lse::DataConfig& data_config,
        const lse::TrainConfig& train_config,
        RNG* const rng,
        const IdentifiersMapT* const identifiers_map) {
    CHECK(!data_config.repository_path().empty());

    return wrap_source_async(train_config, construct_indri_source(data_config, train_config, rng));
}

template <>
DataSource<typename EntityEntity::Objective::BatchType>*
construct_data_source<EntityEntity::Objective>(
        const lse::DataConfig& data_config,
        const lse::TrainConfig& train_config,
        RNG* const rng,
        const IdentifiersMapT* const identifiers_map) {
    CHECK_NE(identifiers_map, (IdentifiersMapT*) nullptr);

    DataSource<typename EntityEntity::Objective::BatchType>* entity_entity_source;

    entity_entity_source = new EntityEntity::DataSource(
        data_config.similarity_path(),
        *identifiers_map,
        rng);

    return new RepeatingSource<EntityEntity::Batch>(
        -1, /* num_repeats */
        entity_entity_source);
}

template <>
DataSource<typename TermTerm::Objective::BatchType>*
construct_data_source<TermTerm::Objective>(
        const lse::DataConfig& data_config,
        const lse::TrainConfig& train_config,
        RNG* const rng,
        const IdentifiersMapT* const identifiers_map) {
    CHECK_NE(identifiers_map, (IdentifiersMapT*) nullptr);

    DataSource<typename TermTerm::Objective::BatchType>* term_term_source =
        new TermTerm::DataSource(
            data_config.similarity_path(),
            *identifiers_map,
            rng);

    return new RepeatingSource<TermTerm::Batch>(
        -1, /* num_repeats */
        term_term_source);
}

template <>
DataSource<typename TextEntityEntityEntity::Objective::BatchType>*
construct_data_source<TextEntityEntityEntity::Objective>(
        const lse::DataConfig& data_config,
        const lse::TrainConfig& train_config,
        RNG* const rng,
        const IdentifiersMapT* const identifiers_map_unused) {
    CHECK(!data_config.repository_path().empty());
    CHECK(!data_config.similarity_path().empty());

    auto indri_source = construct_indri_source(data_config, train_config, rng);

    std::unique_ptr<IdentifiersMapT> identifiers_map(
        indri_source->build_document_identifiers_map());

    DataSource<TextEntity::Batch>* text_entity_source =
        wrap_source_async(train_config, indri_source);

    DataSource<RepresentationSimilarity::Batch>* entity_entity_source =
        construct_data_source<EntityEntity::Objective>(
            data_config, train_config, rng,
            identifiers_map.get());

    return new MultiSource<TextEntity::Batch, EntityEntity::Batch>(
        std::make_tuple(text_entity_source, entity_entity_source));
}

template <>
DataSource<typename TextEntityTermTerm::Objective::BatchType>*
construct_data_source<TextEntityTermTerm::Objective>(
        const lse::DataConfig& data_config,
        const lse::TrainConfig& train_config,
        RNG* const rng,
        const IdentifiersMapT* const identifiers_map_unused) {
    CHECK(!data_config.repository_path().empty());
    CHECK(!data_config.similarity_path().empty());

    auto indri_source = construct_indri_source(data_config, train_config, rng);

    std::unique_ptr<std::map<std::string, WordIdxType>> identifiers_map(
        indri_source->build_term_identifiers_map());

    DataSource<TextEntity::Batch>* text_entity_source =
        wrap_source_async(train_config, indri_source);

    DataSource<TermTerm::Batch>* term_term_source =
         construct_data_source<TermTerm::Objective>(
            data_config, train_config, rng,
            identifiers_map.get());

    return new MultiSource<TextEntity::Batch, TermTerm::Batch>(
        std::make_tuple(text_entity_source, term_term_source));
}

template <typename ObjectiveT>
class DumpModelFn {
 public:
  explicit DumpModelFn(const size_t epoch, const Model<ObjectiveT>* const model)
          : epoch_(epoch), model_(model) {
      CHECK_GE(epoch_, 0);
  }

  void operator()(const std::string& identifier) const {
      if (!FLAGS_output.empty()) {
          std::stringstream ss;
          ss << FLAGS_output << "_" << epoch_;

          if (!identifier.empty()) {
              ss << "_" << identifier;
          }

          ss << ".hdf5";

          const std::string filename = ss.str();

          write_to_hdf5(*model_, filename);
          LOG(INFO) << "Saved model to " << filename << ".";
      }
  }

 private:
  const size_t epoch_;
  const Model<ObjectiveT>* const model_;
};

template <typename ObjectiveT>
std::pair<size_t, LSE::FloatT> iterate_data(const lse::TrainConfig& train_config,
                                            const bool backpropagate,
                                            Model<ObjectiveT>* const model,
                                            DataSource<typename ObjectiveT::BatchType>* const data_source,
                                            typename ObjectiveT::BatchType* const batch,
                                            RNG* const rng,
                                            const DumpModelFn<ObjectiveT>* const dump_model_fn = nullptr) {
    size_t epoch_num_batches = 0;
    LSE::FloatT agg_cost = 0.0;

    const std::chrono::time_point<std::chrono::steady_clock> iteration_start =
        std::chrono::steady_clock::now();

    while (data_source->has_next()) {
        const std::chrono::time_point<std::chrono::steady_clock> batch_start =
            std::chrono::steady_clock::now();

        BatchHandler<typename ObjectiveT::BatchType>::clear(*batch);

        nvtxRangePush("Batch");

        nvtxRangePush("FetchData");
        data_source->next(batch);
        nvtxRangePop();

        const int64 max_threads_per_block =
            Runtime<FLOATING_POINT_TYPE>::getInstance()->props().maxThreadsPerBlock;

        if (BatchHandler<typename ObjectiveT::BatchType>::num_instances(*batch) % max_threads_per_block != 0) {
            LOG(ERROR) << "Skipping Batch #" << epoch_num_batches
                       << " as it is not a multiple of " << max_threads_per_block << " "
                       << "(" << BatchHandler<typename ObjectiveT::BatchType>::num_instances(*batch) << " instances).";
        } else {
            // Save RNG state at beginning of epoch.
            std::stringstream rng_state;
            rng_state << *rng;

            nvtxRangePush("ComputeCost");
            std::unique_ptr<typename ObjectiveT::ForwardResultType> result(
                model->compute_cost(*batch, rng));
            nvtxRangePop();

            nvtxRangePush("ComputeGradients");
            std::unique_ptr<typename ObjectiveT::GradientsType> gradients(
                model->compute_gradients(*result));
            nvtxRangePop();

            if (FLAGS_check_gradients) {
                CHECK(GradientCheckFn<Model<ObjectiveT>>()(
                        model,
                        *batch,
                        *result,
                        *gradients,
                        1e-4 /* epsilon */,
                        1e-1 /* relative_error_threshold */,
                        rng_state,
                        rng))
                    << "Gradient check failed.";
            }

            if (backpropagate) {
                nvtxRangePush("UpdateParameters");
                model->update(*gradients, train_config.learning_rate(), result->scaled_regularization_lambda());
                nvtxRangePop();
            }

            std::chrono::duration<float64> epoch_diff = std::chrono::steady_clock::now() - iteration_start;
            const float64 epoch_duration_until_now = epoch_diff.count();

            std::chrono::duration<float64> batch_diff = std::chrono::steady_clock::now() - batch_start;
            const float64 batch_duration_until_now = batch_diff.count();

            const float64 progress = data_source->progress();

            const float64 seconds_remaining = (
                (1.0 - progress) * (epoch_duration_until_now / progress));

            agg_cost += result->get_cost();
            VLOG(1) << "Batch #" << epoch_num_batches
                    << " ("
                    << std::setprecision(8) << progress * 100.0 << "%; "
                    << seconds_to_humanreadable_time(seconds_remaining) << " remaining"
                    << "): "
                    << "cost=" << result->get_cost() << ", "
                    << "duration=" << batch_duration_until_now;
        }

        if (dump_model_fn != nullptr &&
                FLAGS_dump_every > 0 &&
                epoch_num_batches > 0 &&
                epoch_num_batches % FLAGS_dump_every == 0) {
            (*dump_model_fn)(std::to_string(epoch_num_batches));
        }

        ++ epoch_num_batches;

        nvtxRangePop();
    }

    CHECK_GT(epoch_num_batches, 0) << "No batches to train during epoch";

    return std::make_pair(epoch_num_batches, agg_cost);
}

template <typename T>
T ParseProto(const std::string& msg) {
    T proto;
    CHECK(google::protobuf::TextFormat::ParseFromString(msg, &proto));

    return proto;
}

const std::map<std::string, UpdateMethodConf> UPDATE_METHODS {
    {"sgd", ParseProto<UpdateMethodConf>("type: SGD")},
    {"adagrad", ParseProto<UpdateMethodConf>("type: ADAGRAD")},
    {"sparse_adam", ParseProto<UpdateMethodConf>("type: ADAM adam_conf: < mode: SPARSE >")},
    {"dense_adam", ParseProto<UpdateMethodConf>("type: ADAM adam_conf: < mode: DENSE_UPDATE >")},
    {"full_adam", ParseProto<UpdateMethodConf>("type: ADAM adam_conf: < mode: DENSE_UPDATE_DENSE_VARIANCE >")},
};

const std::map<std::string, lse::ModelDesc::TransformDesc::Nonlinearity> NONLINEARITIES {
    {"tanh", lse::ModelDesc::TransformDesc::TANH},
    {"hard_tanh", lse::ModelDesc::TransformDesc::HARD_TANH},
};

template <typename ObjectiveT>
void train(const lse::ModelDesc& model_desc,
           const lse::DataConfig& data_config,
           const lse::TrainConfig& train_config,
           RNG* rng) {
    std::unique_ptr<DataSource<typename ObjectiveT::BatchType>> data_source(
        construct_data_source<ObjectiveT>(
            data_config, train_config, rng));

    // Extract meta data through a generic interface.
    lse::Metadata meta;
    data_source->extract_metadata(&meta);

    const size_t vocabulary_size = meta.term_size();
    const size_t corpus_size = meta.object_size();

    CHECK_GT(vocabulary_size, 0);
    CHECK_GT(corpus_size, 0);

    LOG(INFO) << "Training statistics: "
              << "vocabulary size=" << vocabulary_size << ", "
              << "corpus size=" << corpus_size;

    Model<ObjectiveT> model(vocabulary_size,
                            corpus_size,
                            model_desc,
                            train_config);

    model.initialize(rng);

    cudaDeviceSynchronize();

    LOG(INFO) << "Initialized LSE with " << model.num_parameters() << " parameters "
              << "for training on " << vocabulary_size << " words and " << corpus_size << " objects.";

    if (!FLAGS_output.empty()) {
        std::stringstream ss;
        ss << FLAGS_output << "_meta";

        const std::string filename = ss.str();

        std::ofstream meta_file;
        meta_file.open(filename);

        meta.SerializeToOstream(&meta_file);
    }

    std::unique_ptr<typename ObjectiveT::BatchType> batch(
        BatchHandler<typename ObjectiveT::BatchType>::create(train_config));

    std::vector<LSE::FloatT> epoch_costs;

    if (FLAGS_compute_initial_cost) {
        size_t epoch_num_batches;
        LSE::FloatT agg_cost;

        std::tie(epoch_num_batches, agg_cost) = iterate_data<ObjectiveT>(
            train_config,
            false, /* backpropagate */
            &model,
            data_source.get(),
            batch.get(),
            rng);

        data_source->reset();

        const LSE::FloatT initial_cost = agg_cost / epoch_num_batches;
        epoch_costs.push_back(initial_cost);

        LOG(INFO) << "Epoch #0 (initial): cost=" << epoch_costs;
    }

    if (FLAGS_dump_initial_model) {
        DumpModelFn<ObjectiveT> dump_model_fn(0, &model);

        // Dump model.
        dump_model_fn("" /* empty identifier */);
    }

    const std::chrono::time_point<std::chrono::steady_clock> start =
        std::chrono::steady_clock::now();

    size_t num_batches = 0;

    for (size_t epoch = 1; epoch <= train_config.num_epochs(); ++epoch) {
        const std::chrono::time_point<std::chrono::steady_clock> epoch_start =
            std::chrono::steady_clock::now();

        DumpModelFn<ObjectiveT> dump_model_fn(epoch, &model);

        nvtxRangePush("Epoch");

        size_t epoch_num_batches;
        LSE::FloatT agg_cost;

        std::tie(epoch_num_batches, agg_cost) = iterate_data<ObjectiveT>(
            train_config,
            true, /* backpropagate */
            &model,
            data_source.get(),
            batch.get(),
            rng,
            &dump_model_fn);

        num_batches += epoch_num_batches;

        const std::chrono::duration<float64> epoch_duration =
            std::chrono::steady_clock::now() - epoch_start;

        const std::chrono::duration<float64> total_duration =
            std::chrono::steady_clock::now() - start;

        const float64 batches_per_second = num_batches / total_duration.count();

        const LSE::FloatT epoch_cost = agg_cost / epoch_num_batches;
        epoch_costs.push_back(epoch_cost);

        LOG(INFO) << "Epoch #" << epoch << ": "
                  << "duration=" << seconds_to_humanreadable_time(epoch_duration.count()) << " "
                  << "(" << batches_per_second << " batches/second) "
                  << "cost=" << epoch_costs;

        // Dump model.
        dump_model_fn("" /* empty identifier */);

        data_source->reset();

        nvtxRangePop();
    }
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::set_terminate(exception_handler);

    CHECK_GE(argc, 2) << "Usage: " << argv[0] << " [OPTIONS] <path to Indri index>";

    CHECK(contains_key(UPDATE_METHODS, FLAGS_update_method))
        << "Please specify a valid --update_method.";

    CHECK(contains_key(WEIGHTING_STRATEGIES, FLAGS_weighting))
        << "Please specify a valid --weighting.";

    // Model.
    lse::ModelDesc model_desc;
    model_desc.set_word_repr_size(FLAGS_word_repr_size);
    model_desc.set_entity_repr_size(FLAGS_entity_repr_size);

    model_desc.mutable_transform_desc()->set_batch_normalization(FLAGS_batch_normalization);
    model_desc.mutable_transform_desc()->set_nonlinearity(NONLINEARITIES.at(FLAGS_nonlinearity));

    model_desc.set_clip_sigmoid(true);
    model_desc.set_bias_negative_samples(FLAGS_bias_negative_samples);

    model_desc.set_l2_normalize_phrase_reprs(FLAGS_l2_phrase_normalization);
    model_desc.set_l2_normalize_entity_reprs(FLAGS_l2_entity_normalization);

    // Data.
    lse::DataConfig data_config;

    const std::string repository_path = argv[1];
    data_config.set_repository_path(repository_path);

    if (argc >= 3) {
        const std::string similarity_path = argv[2];
        data_config.set_similarity_path(similarity_path);
    }

    data_config.set_max_vocabulary_size(FLAGS_max_vocabulary_size);
    data_config.set_min_document_frequency(FLAGS_min_document_frequency);

    uint64 max_document_frequency = 0;

    if (FLAGS_max_document_frequency <= 1.0) {
        max_document_frequency = static_cast<uint64>(
            ceil(indri::GetDocumentCount(repository_path) * FLAGS_max_document_frequency));

        LOG(INFO) << "Setting max_document_frequency to " << max_document_frequency << ".";
    } else {
        max_document_frequency = static_cast<uint64>(FLAGS_max_document_frequency);
    }

    data_config.set_max_document_frequency(max_document_frequency);

    data_config.set_include_oov(FLAGS_include_oov);

    CHECK_GT(data_config.max_vocabulary_size(), 0);
    CHECK_GE(data_config.max_document_frequency(), 0);

    // Training.
    lse::TrainConfig train_config;
    train_config.set_num_epochs(FLAGS_num_epochs);
    train_config.set_batch_size(FLAGS_batch_size);

    train_config.set_window_size(FLAGS_window_size);
    train_config.set_num_random_entities(FLAGS_num_random_entities);

    train_config.set_regularization_lambda(FLAGS_regularization_lambda);
    train_config.set_learning_rate(FLAGS_learning_rate);

    train_config.mutable_update_method()->CopyFrom(UPDATE_METHODS.at(FLAGS_update_method));

    train_config.set_no_shuffle(FLAGS_no_shuffle);

    CHECK_GE(FLAGS_entity_similarity_weight, 0.0);
    CHECK_LE(FLAGS_entity_similarity_weight, 1.0);

    CHECK_GE(FLAGS_term_similarity_weight, 0.0);
    CHECK_LE(FLAGS_term_similarity_weight, 1.0);

    train_config.set_text_entity_weight(1.0 - FLAGS_entity_similarity_weight - FLAGS_term_similarity_weight);
    train_config.set_entity_entity_weight(FLAGS_entity_similarity_weight);
    train_config.set_term_term_weight(FLAGS_term_similarity_weight);

    CHECK_GT(FLAGS_seed, 0) << "Please specify a --seed value.";

    if (train_config.learning_rate() == 0.0) {
        switch (train_config.update_method().type()) {
            default:
            case SGD:
            case ADAGRAD:
                train_config.set_learning_rate(0.01);
                break;
            case ADAM:
                train_config.set_learning_rate(0.001);
                break;
        }
    }

    LOG(INFO) << "Model descriptor: " << model_desc;
    LOG(INFO) << "Data configuration: " << data_config;
    LOG(INFO) << "Training configuration: " << train_config;

    LOG(INFO) << SHOW_DEFINE(FLOATING_POINT_TYPE);

    RNG rng;
    rng.seed(FLAGS_seed);


    if (train_config.entity_entity_weight() != 0.0) {
        CHECK(!data_config.similarity_path().empty());
        CHECK_EQ(train_config.term_term_weight(), 0.0);

        train<TextEntityEntityEntity::Objective>(
            model_desc,
            data_config,
            train_config,
            &rng);
    } else if (train_config.term_term_weight() != 0.0) {
        CHECK(!data_config.similarity_path().empty());
        CHECK_EQ(train_config.entity_entity_weight(), 0.0);

        train<TextEntityTermTerm::Objective>(
            model_desc,
            data_config,
            train_config,
            &rng);
    } else {
        train<TextEntity::Objective>(
            model_desc,
            data_config,
            train_config,
            &rng);
    }

    // Synchronize GPU.
    cudaDeviceSynchronize();

    // Notify profiler that we are done.
    cudaProfilerStop();

    DLOG(INFO) << "Finished.";

    return 0;
}