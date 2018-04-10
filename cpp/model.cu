#include "cuNVSM/model.h"

#include <algorithm>
#include <glog/logging.h>

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
ModelBase<FloatT, WordIdxType, EntityIdxType>::ModelBase(
        const size_t num_words,
        const size_t num_entities,
        const lse::ModelDesc& desc,
        const lse::TrainConfig& train_config)
    : desc_(desc),
      streams_(new DefaultStream), // Multiple streams do not seem to improve training speed,
                                   // but creates issues with memory allocation.
      words_(WORD_REPRS,
             num_words, desc_.word_repr_size(),
             train_config.update_method(),
             streams_.get()),
      entities_(ENTITY_REPRS,
                num_entities, desc_.entity_repr_size(),
                train_config.update_method(),
                streams_.get()),
      transform_(TRANSFORM,
                 desc_.transform_desc(),
                 desc_.word_repr_size(), desc_.entity_repr_size(),
                 train_config.update_method(),
                 streams_.get()),
      params_({
          {WORD_REPRS, &words_},
          {ENTITY_REPRS, &entities_},
          {TRANSFORM, &transform_},
      }) {}

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
ModelBase<FloatT, WordIdxType, EntityIdxType>::~ModelBase() {}

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
void ModelBase<FloatT, WordIdxType, EntityIdxType>::initialize(
        RNG* const rng) {
    words_.initialize(rng);
    entities_.initialize(rng);
    transform_.initialize(rng);
}

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
device_matrix<FloatT>*
ModelBase<FloatT, WordIdxType, EntityIdxType>::get_phrase_representations(
        const cudaStream_t stream,
        const device_matrix<WordIdxType>& flattened_words,
        const size_t window_size,
        const device_matrix<FloatT>* const flattened_word_weights) const {
    DCHECK(initialized());
    DCHECK_EQ(flattened_words.size() % window_size, 0);

    device_matrix<FloatT>* const phrase_reprs =
        words_.get_average_representations(stream, flattened_words, window_size,
                                           flattened_word_weights);

    CHECK_MATRIX(*phrase_reprs);

    return phrase_reprs;
}

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
typename Storage<FloatT>::DataType
ModelBase<FloatT, WordIdxType, EntityIdxType>::get_data() const {
    DCHECK(this->initialized());

    typename Storage<FloatT>::DataType data;

    for (const auto& pair : params_) {
        const ParamIdentifier param_id = pair.first;
        Parameters<FloatT>* const param = pair.second;

        Storage<FloatT>* const storage = dynamic_cast<Storage<FloatT>*>(param);
        CHECK_NOTNULL(storage);

        for (const auto& matrix_pair : storage->get_data()) {
            std::stringstream ss;
            ss << ParamName[param_id] << "-" << matrix_pair.first;

            const std::string name = ss.str();

            const device_matrix<FloatT>* matrix = matrix_pair.second;

            CHECK(!contains_key(data, name));

            data.insert(std::make_pair(name, matrix));
        }
    }

    return data;
}

template <typename ObjectiveT>
Model<ObjectiveT>::Model(
        const size_t num_words,
        const size_t num_entities,
        const lse::ModelDesc& desc,
        const lse::TrainConfig& train_config)
    : ModelBase<FloatT, WordIdxType, EntityIdxType>(
          num_words, num_entities, desc, train_config),
      objective_(new ObjectiveT(this, train_config)) {}

template <typename ObjectiveT>
device_matrix<typename Model<ObjectiveT>::FloatT>* Model<ObjectiveT>::infer(
        const std::vector<std::vector<WordIdxType> >& words,
        const size_t window_size) const {
    PROFILE_FUNCTION();
    DCHECK(this->initialized());

    const cudaStream_t stream = DefaultStream::get()->next();

    std::unique_ptr<device_matrix<WordIdxType>> flattened_words(
        new device_matrix<WordIdxType>(words.size() * window_size, 1, stream));
    flatten(stream, words, flattened_words.get());

    // Get phrase representations.
    std::unique_ptr<device_matrix<FloatT> > phrase_reprs(
        this->get_phrase_representations(
            stream, *flattened_words, window_size));

    // Project to entity space.
    std::unique_ptr<device_matrix<FloatT> > word_repr_projections(
        this->transform_.transform(
            stream,
            *phrase_reprs,
            nullptr /* batch_normalization */));

    cudaDeviceSynchronize();

    return word_repr_projections.release();
}

template <typename ObjectiveT>
typename Model<ObjectiveT>::ForwardResult*
Model<ObjectiveT>::compute_cost(
        const Batch& batch,
        RNG* const rng) const {
    DCHECK(this->initialized());

    return objective_->compute_cost(batch, rng);
}

template <typename ObjectiveT>
typename Model<ObjectiveT>::Gradients*
Model<ObjectiveT>::compute_gradients(
        const ForwardResult& result) {
    DCHECK(this->initialized());

    return objective_->compute_gradients(result);
}

template <typename ObjectiveT>
typename Model<ObjectiveT>::FloatT
Model<ObjectiveT>::get_cost(
        const Batch& batch,
        const std::stringstream* const rng_state,
        RNG* const rng) const {
    PROFILE_FUNCTION();
    DCHECK(this->initialized());

    if (rng_state != nullptr) {
        CHECK(!rng_state->eof() && rng_state->good());
        std::stringstream rng_state_copy;
        rng_state_copy << rng_state->str();

        rng_state_copy >> *rng;
    }

    std::unique_ptr<ForwardResult> result(compute_cost(batch, rng));

    return result->get_cost();
}

template <typename ObjectiveT>
void Model<ObjectiveT>::backprop(
        const ForwardResult& result,
        const FloatT learning_rate) {
    PROFILE_FUNCTION();
    DCHECK(this->initialized());

    std::unique_ptr<Gradients> gradients(compute_gradients(result));
    update(*gradients, learning_rate, result.scaled_regularization_lambda());
}

template <typename ObjectiveT>
void Model<ObjectiveT>::update(
        const Gradients& gradients,
        const FloatT learning_rate,
        const FloatT scaled_regularization_lambda) {
    DCHECK(this->initialized());

    //
    // Apply gradients.
    //

    // TODO(cvangysel): make async using streams.

    CHECK_MATRIX_NORM(*gradients.grad_entity_repr_);

    // Update entity representations.
    this->entities_.update(
        gradients, learning_rate, scaled_regularization_lambda,
        this->streams_.get());

    CHECK_MATRIX_NORM(*gradients.grad_phrase_reprs_);

    // Update word representations.
    this->words_.update(
        gradients, learning_rate, scaled_regularization_lambda,
        this->streams_.get());

    CHECK_MATRIX_NORM(*gradients.grad_transform_matrix_);
    CHECK_MATRIX_NORM(*gradients.grad_bias_);

    this->transform_.update(
        gradients, learning_rate, scaled_regularization_lambda,
        this->streams_.get());
}

// Explicit instantiations.
template class ModelBase<FLOATING_POINT_TYPE, int32, int32>;
template class Model<TextEntity::Objective>;
template class Model<EntityEntity::Objective>;
template class Model<TermTerm::Objective>;
template class Model<TextEntityEntityEntity::Objective>;
template class Model<TextEntityTermTerm::Objective>;
