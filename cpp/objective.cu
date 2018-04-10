#include "cuNVSM/model.h"

namespace TextEntity {

void Objective::generate_labels(
        const EntityIdxType* const labels,
        const size_t num_labels,
        const size_t num_negative_labels,
        std::vector<EntityIdxType>* const instance_entities,
        RNG* const rng) const {
    PROFILE_FUNCTION();

    CHECK(instance_entities->empty());

    const size_t num_repeats = num_negative_labels + 1;
    const size_t total_labels = num_labels * num_repeats;

    instance_entities->resize(total_labels, 0);

    label_generator_->generate(
        labels, model_->entities_,
        num_labels, num_negative_labels,
        instance_entities, /* dst */
        rng);

    DCHECK_EQ(instance_entities->size(), total_labels);
    CHECK_EQ(instance_entities->size() % num_repeats, 0);
}

Objective::ForwardResultType* Objective::compute_cost(
        const Batch& batch, RNG* const rng) const {
    PROFILE_FUNCTION();

    const size_t num_random_entities = train_config_.num_random_entities();

    std::unique_ptr<device_matrix<WordIdxType>> instance_words(
        device_matrix<WordIdxType>::create(
            model_->streams_->next(),
            batch.features_, /* begin */
            batch.features_+ batch.num_instances_ * batch.window_size(), /* end */
            1, /* num_rows */
            batch.num_instances_ * batch.window_size() /* num_cols */));

    std::unique_ptr<device_matrix<FloatT>> instance_word_weights(
        device_matrix<FloatT>::create(
            model_->streams_->next(),
            batch.feature_weights_, /* begin */
            batch.feature_weights_+ batch.num_instances_ * batch.window_size(), /* end */
            1, /* num_rows */
            batch.num_instances_ * batch.window_size() /* num_cols */));

    // Transfer instance weights to device.
    //
    // TODO(cvangysel): maybe these weights can live on the GPU?
    std::unique_ptr<device_matrix<FloatT>> instance_weights(
        device_matrix<FloatT>::create(
            model_->streams_->next(),
            batch.weights_, /* begin */
            batch.weights_+ batch.num_instances_, /* end */
            1, /* num_rows */
            batch.num_instances_ /* num_cols */));

    // Figure out the entities in the sample.
    const size_t num_entities = batch.num_instances_;
    const size_t num_repeats = num_random_entities + 1;
    const size_t total_entity_ids = num_repeats * num_entities;

    std::unique_ptr<device_matrix<WordIdxType>> instance_entities(
        new device_matrix<WordIdxType>(
            1, /* num_rows */
            total_entity_ids /* num_cols */,
            model_->streams_->next()));

    // Generate identifiers for target and negative entities.
    {
        std::vector<EntityIdxType> tmp_instance_entities;

        generate_labels(batch.labels_,
                        num_entities, /* num_entities */
                        num_random_entities, /* num_negative_labels */
                        &tmp_instance_entities,
                        rng);

        CHECK_EQ(tmp_instance_entities.size(), total_entity_ids);

        instance_entities->fillwith(
            instance_entities->getStream(),
            tmp_instance_entities);
    }

    std::unique_ptr<ForwardResultType> result(
        new ForwardResultType(instance_words.release(),
                              instance_word_weights.release(),
                              instance_entities.release(),
                              batch.window_size(),
                              num_random_entities,
                              train_config_.regularization_lambda()));

    if (model_->desc_.l2_normalize_phrase_reprs()) {
        result->phrase_normalizer_.reset(new Normalizer<FloatT>(
            num_entities /* num_instances */));
    }

    if (model_->desc_.l2_normalize_entity_reprs()) {
        result->entity_normalizer_.reset(new Normalizer<FloatT>(
            total_entity_ids /* num_instances */));
    }

    if (model_->desc_.transform_desc().batch_normalization()) {
        result->batch_normalization_.reset(
            new BatchNormalization<FloatT>(
                model_->desc_.entity_repr_size(), /* num_features */
                0.1, /* momentum */
                1e-4, /* epsilon */
                true /* cache_input */));
    }

    DCHECK_EQ(instance_weights->size(), result->batch_size_);
    DCHECK_EQ(result->entity_ids_->size() / num_repeats, result->batch_size_);

    //
    // From word representations to projections in entity space.
    //

    // Get phrase representations.
    result->phrase_reprs_.reset(
        model_->get_phrase_representations(result->flattened_words_->getStream(),
                                           *result->flattened_words_,
                                           batch.window_size(),
                                           result->flattened_word_weights_.get()));

    CHECK_DIMENSIONS((*result->phrase_reprs_),
                     model_->words_.size(), num_entities);
    CHECK_MATRIX(*result->phrase_reprs_);

    if (result->phrase_normalizer_ != nullptr) {
        result->phrase_normalizer_->forward(
            *result->phrase_reprs_, /* input */
            result->phrase_reprs_.get() /* output */);

        CHECK_MATRIX(*result->phrase_reprs_);
    }

    // Project to entity space.
    result->word_projections_.reset(
        model_->transform_.transform(result->phrase_reprs_->getStream(),
                                   *result->phrase_reprs_,
                                   result->batch_normalization_.get()));

    CHECK_DIMENSIONS(*result->word_projections_,
                     model_->entities_.size(), num_entities);
    CHECK_MATRIX(*result->word_projections_);

    //
    // From projections to NCE cost.
    //

    // Broadcast projections.
    result->broadcasted_word_projections_.reset(
        broadcast_columns(result->word_projections_->getStream(),
                          *result->word_projections_, num_repeats));

    // Fetch entity representations.
    result->entity_representations_.reset(
        model_->entities_.get_representations(
            model_->streams_->next(), *result->entity_ids_));

    if (result->entity_normalizer_ != nullptr) {
        result->entity_normalizer_->forward(
            *result->entity_representations_,
            result->entity_representations_.get());

        CHECK_MATRIX(*result->entity_representations_);
    }

    // Negate the representations belonging to negative instances;
    // this works for us both in the forward as in the backward passes.
    //
    // Forward pass:
    //  sigmoid(-x) = 1.0 - sigmoid(x)
    //
    // and such, we can back propagate the errors without special casing
    // the negative instances.
    apply_except_every_Nth_column<thrust::negate<FloatT>>(
        thrust::cuda::par.on(result->entity_representations_->getStream()),
        num_random_entities + 1 /* col_idx */,
        result->entity_representations_.get());

    CHECK_MATRIX(*result->entity_representations_);
    CHECK_MATRIX(*result->broadcasted_word_projections_);

    CHECK_DIMENSIONS_EQUAL(*result->entity_representations_,
                           *result->broadcasted_word_projections_);

    // Hadamard product.
    device_matrix<FloatT> multiplied_representations(
        model_->entities_.size(), total_entity_ids,
        model_->streams_->next());
    MAKE_MATRIX_NULL(multiplied_representations);

    CHECK_DIMENSIONS_EQUAL(multiplied_representations,
                           *result->entity_representations_);

    // Merge streams.
    const cudaStream_t words_and_entities_stream = merge_streams(
        result->broadcasted_word_projections_->getStream(),
        merge_streams(result->entity_representations_->getStream(),
                      multiplied_representations.getStream()));

    {
        thrust::transform(
            thrust::cuda::par.on(words_and_entities_stream),
            begin(*result->broadcasted_word_projections_), /* first op */
            end(*result->broadcasted_word_projections_),
            begin(*result->entity_representations_), /* second op */
            begin(multiplied_representations), /* dest */
            thrust::multiplies<FloatT>());

        CHECK_MATRIX(multiplied_representations);
    }

    // Compute the similarity probabilities (for every entity, either positive or negative).
    result->similarity_probs_.reset(
        new device_matrix<FloatT>(1, total_entity_ids,
                                  model_->streams_->next()));
    CHECK_EQ(total_entity_ids, multiplied_representations.getCols());

    const cudaStream_t probs_and_words_and_entities_stream = merge_streams(
        result->similarity_probs_->getStream(),
        words_and_entities_stream);

    {
        // Aggregate multiplied representations (i.e. finish the dot product);
        // reduce_axis does not expect nulled output.
        reduce_axis(
            probs_and_words_and_entities_stream,
            FIRST_AXIS,
            multiplied_representations,
            result->similarity_probs_.get());

        // Apply sigmoid, clipped between epsilon and 1.0 - epsilon.
        apply_elemwise<func::truncated_sigmoid<FloatT>>(
            thrust::cuda::par.on(probs_and_words_and_entities_stream),
            result->similarity_probs_.get(),
            func::truncated_sigmoid<FloatT>(
                model_->desc_.clip_sigmoid() ? 1e-7 : 0.0 /* epsilon */));
    }

    // Make a copy for the log_probs.
    result->pointwise_mass_.reset(
        result->similarity_probs_->copy(probs_and_words_and_entities_stream));

    // Convert to log-probs.
    apply_elemwise<func::log<FloatT> >(
        thrust::cuda::par.on(probs_and_words_and_entities_stream),
        result->pointwise_mass_.get());

    // For every positive example, we sample one or more negative classes;
    // this introduces an artificial bias towards negative classes.
    //
    // Given that we sample an equal amount of instances from every document,
    // every class receives the same number of updates. However, as the introduced
    // bias prefers negative classes, we postulate that this causes all documents
    // to live very close near each other in a restricted area of the space.
    //
    // Most likely, if you train long enough with this bias enabled; the learning
    // process figures it out.
    if (!model_->desc_.bias_negative_samples() && num_random_entities > 1) {
        // Reweights everything such that the cost function remains const
        instance_weights->scale(
            instance_weights->getStream(),
            (static_cast<FloatT>(num_random_entities) + 1.0) /
            (2.0 * static_cast<FloatT>(num_random_entities)));
    }

    // Broadcast instance weights.
    result->broadcasted_instance_weights_.reset(
        broadcast_columns(instance_weights->getStream(),
                          *instance_weights, num_repeats));

    // Continuation of bias correction above.
    if (!model_->desc_.bias_negative_samples() && num_random_entities > 1) {
        // Upweights the positive instances; every positive instances becomes
        // equal to the number of negative instances.
        apply_every_Nth_column(
            thrust::cuda::par.on(result->broadcasted_instance_weights_->getStream()),
            num_repeats /* col_idx */,
            result->broadcasted_instance_weights_.get(),
            func::scale_by_constant<FloatT>(num_random_entities));
    }

    // Verify dimensions.
    CHECK_DIMENSIONS_EQUAL(*result->pointwise_mass_, *result->broadcasted_instance_weights_);

    const cudaStream_t predictions_and_weights_stream = merge_streams(
        probs_and_words_and_entities_stream,
        result->broadcasted_instance_weights_->getStream());

    {
        // Again, Hadamard, but now between point-wise contributions and their weights.
        hadamard_product(
            thrust::cuda::par.on(predictions_and_weights_stream),
            *result->broadcasted_instance_weights_,
            result->pointwise_mass_.get());
    }

    // Synchronize here, as the gradient phase has not been fully stream-lined (pun intended).
    // streams_->synchronize();

    DCHECK(result->complete());

    return result.release();
}

Objective::GradientsType* Objective::compute_gradients(const ForwardResultType& result) {
    PROFILE_FUNCTION();

    DCHECK(result.complete());

    const cudaStream_t stream = model_->streams_->next();

    std::unique_ptr<GradientsType> gradients(new SingleGradients<::Typedefs::FloatT>(&result));

    // As we take the negative of the joint-log probability and consequently
    // subtract the derivative of this quantity w.r.t. the weights, we can
    // simply add the derivative (i.e. no need to negate, and we do gradient ascent).

    const size_t num_entities = result.entity_ids_->size();

    // Broadcast the phrase representations.
    const size_t num_repeats = (result.num_random_entities_ + 1);
    CHECK_EQ(num_repeats * result.batch_size_, num_entities);

    // Keep track of multipliers for every entity instance.
    device_matrix<FloatT> instance_multipliers(
        1 /* num rows */, num_entities /* num columns*/,
        model_->streams_->next());

    const cudaStream_t instance_multipliers_stream = merge_streams(
        instance_multipliers.getStream(),
        merge_streams(result.broadcasted_instance_weights_->getStream(),
                      result.similarity_probs_->getStream()));

    CHECK_DIMENSIONS_EQUAL(
        instance_multipliers,
        (*result.broadcasted_instance_weights_));

    CHECK_DIMENSIONS_EQUAL(
        instance_multipliers,
        (*result.similarity_probs_));

    // Multiply learning rate divided by batch size with
    // the per-instance weights.
    const FloatT batch_size_normalizer = exp(-log(result.batch_size_));

    // Multiply with the complement of the probabilities.
    thrust::transform(
        thrust::cuda::par.on(instance_multipliers_stream),
        // Multiply in instance weights.
        begin(*result.broadcasted_instance_weights_),
        end(*result.broadcasted_instance_weights_), /* first op*/
        // Multiply in batch normalization constant.
        make_scalar_multiplication_iterator(
            // Multiply in similarity probabilities derivatives.
            thrust::make_transform_iterator(
                begin(*result.similarity_probs_),
                func::sigmoid_to_log_sigmoid_deriv<FloatT>(
                    model_->desc_.clip_sigmoid() ? 1e-6 : 0.0 /* epsilon */)),
            batch_size_normalizer), /* second op */
        begin(instance_multipliers), /* result */
        thrust::multiplies<FloatT>());

    // Get pointers to intermediate results which we will use.
    const device_matrix<FloatT>* const entity_reprs = result.entity_representations_.get();
    CHECK_EQ(entity_reprs->getCols(), num_entities);

    CHECK_DIMENSIONS((*result.broadcasted_word_projections_),
                     model_->entities_.size(), num_entities);

    // d cost / d entity_reprs
    gradients->grad_entity_repr_.reset(
        result.broadcasted_word_projections_->copy(
            result.broadcasted_word_projections_->getStream()));

    // Verify dimensions.
    CHECK_DIMENSIONS_EQUAL((*gradients->grad_entity_repr_),
                           (*result.broadcasted_word_projections_));
    CHECK_EQ(gradients->grad_entity_repr_->getCols(),
             instance_multipliers.getCols());

    // Merge in multipliers.
    apply_columnwise<thrust::multiplies<FloatT>>(
        thrust::cuda::par.on(stream),
        instance_multipliers,
        gradients->grad_entity_repr_.get());

    // Negate the appropriate representations again.
    apply_except_every_Nth_column<thrust::negate<FloatT>>(
        thrust::cuda::par.on(stream),
        num_repeats /* col_idx */,
        gradients->grad_entity_repr_.get());

    CHECK_MATRIX(*gradients->grad_entity_repr_);

    if (result.entity_normalizer_ != nullptr) {
        gradients->grad_entity_repr_.reset(
            result.entity_normalizer_->backward(
                *gradients->grad_entity_repr_));

        CHECK_DIMENSIONS_EQUAL(*gradients->grad_entity_repr_,
                               *result.broadcasted_word_projections_);
    }

    //
    // Back-propagate through projection: d cost / d projection.
    //

    const size_t src_num_cols = result.num_random_entities_ + 1;

    std::unique_ptr<device_matrix<FloatT>> grad_projection(
        fold_columns(
            entity_reprs->getStream(),
            *entity_reprs,
            src_num_cols,
            &instance_multipliers));

    CHECK_DIMENSIONS(*grad_projection,
                     model_->entities_.size(), result.batch_size_);

    gradients->grad_bias_.reset(
        new device_matrix<FloatT>(model_->entities_.size(), 1, stream));
    gradients->grad_transform_matrix_.reset(
        new device_matrix<FloatT>(model_->entities_.size(), model_->words_.size(), stream));

    // Back-propagate through transform layer.
    model_->transform_.backward(
        stream,
        result,
        *result.phrase_reprs_,
        *result.word_projections_,
        grad_projection.get(),
        gradients.get());

    // d cost / d word_reprs
    //
    // (entity_repr_size by word_repr_size)^T X (entity_repr_size by batch_size_)
    gradients->grad_phrase_reprs_.reset(
        new device_matrix<FloatT>(model_->words_.size(),
                                  result.batch_size_,
                                  stream));

    // TODO(cvangysel): this logic should also be moved to Transform::backward.
    matrix_mult(stream,
                model_->transform_.transform_, CUBLAS_OP_T,
                *grad_projection, CUBLAS_OP_N,
                gradients->grad_phrase_reprs_.get());

    CHECK_DIMENSIONS(*gradients->grad_phrase_reprs_,
                     model_->words_.size(), result.batch_size_);

    if (result.phrase_normalizer_ != nullptr) {
        gradients->grad_phrase_reprs_.reset(
            result.phrase_normalizer_->backward(
                *gradients->grad_phrase_reprs_));

        CHECK_DIMENSIONS(*gradients->grad_phrase_reprs_,
                         model_->words_.size(), result.batch_size_);
    }

    // Divide by window_size, as we took the average word representation in the forward pass.
    thrust::transform(thrust::cuda::par.on(stream),
                      gradients->grad_phrase_reprs_->begin(),
                      gradients->grad_phrase_reprs_->end(),
                      gradients->grad_phrase_reprs_->begin(),
                      func::scale_by_constant<FloatT>(
                            exp(-log(result.window_size_)) /* result.window_size_^-1 */));

    CHECK_MATRIX(*gradients->grad_phrase_reprs_);

    return gradients.release();
}

}  // namespace TextEntity

namespace RepresentationSimilarity {

Objective::ForwardResultType* Objective::compute_cost(
        const Batch& batch, RNG* const rng) const {
    PROFILE_FUNCTION();

    const size_t num_instances = batch.num_instances_;
    const size_t total_ids = 2 * num_instances;

    std::unique_ptr<device_matrix<ObjectIdxType>> instance_ids(
        device_matrix<ObjectIdxType>::create(
            model_->streams_->next(),
            batch.features_,
            batch.features_ + total_ids,
            1, /* num_rows */
            total_ids /* num_cols */));

    std::unique_ptr<device_matrix<FloatT>> instance_weights(
        device_matrix<FloatT>::create(
            model_->streams_->next(),
            batch.weights_, /* begin */
            batch.weights_+ num_instances, /* end */
            1, /* num_rows */
            num_instances /* num_cols */));

    print_matrix(*instance_weights);

    std::unique_ptr<ForwardResultType> result(
        new ForwardResultType(param_id_,
                              instance_ids.release(),
                              instance_weights.release(),
                              train_config_.regularization_lambda()));

    // Fetch representations.
    result->representations_.reset(
        get_representation_storage()->get_representations(
            model_->streams_->next(), *result->ids_));

    CHECK_DIMENSIONS(*result->representations_,
                     get_representation_storage()->size(), result->ids_->getCols());

    std::unique_ptr<device_matrix<FloatT>> multiplied_representations(
        fold_columns<FloatT, thrust::multiplies<FloatT>>(
            result->representations_->getStream(),
            *result->representations_,
            2 /* cluster_size */));

    // Compute the similarity probabilities (for every representation, either positive or negative).
    result->similarity_probs_.reset(
        new device_matrix<FloatT>(1, num_instances,
                                  model_->streams_->next()));
    CHECK_EQ(num_instances, multiplied_representations->getCols());

    reduce_axis(merge_streams(
                    multiplied_representations->getStream(),
                    result->similarity_probs_->getStream()),
                FIRST_AXIS,
                *multiplied_representations,
                result->similarity_probs_.get());

    // Apply sigmoid, clipped between epsilon and 1.0 - epsilon.
    apply_elemwise<func::truncated_sigmoid<FloatT>>(
        thrust::cuda::par.on(result->similarity_probs_->getStream()),
        result->similarity_probs_.get(),
        func::truncated_sigmoid<FloatT>(
            model_->desc_.clip_sigmoid() ? 1e-7 : 0.0 /* epsilon */));

    // Make a copy for the log_probs.
    result->pointwise_mass_.reset(
        result->similarity_probs_->copy(result->similarity_probs_->getStream()));

    // Convert to log-probabilities.
    apply_elemwise<func::log<FloatT> >(
        thrust::cuda::par.on(result->pointwise_mass_->getStream()),
        result->pointwise_mass_.get());

    elemwise_binary(
        thrust::cuda::par.on(merge_streams(
            result->pointwise_mass_->getStream(),
            result->weights_->getStream())),
        *result->weights_, /* first op */
        result->pointwise_mass_.get(), /* second op and dst */
        thrust::multiplies<FloatT>());

    // Synchronize here, as the gradient phase has not been fully stream-lined (pun intended).
    // streams_->synchronize();

    DCHECK(result->complete());

    return result.release();
}

Objective::GradientsType* Objective::compute_gradients(const ForwardResultType& result) {
    PROFILE_FUNCTION();

    DCHECK(result.complete());

    const cudaStream_t stream = model_->streams_->next();

    std::unique_ptr<GradientsType> gradients(new SingleGradients<::Typedefs::FloatT>(&result));

    // As we take the negative of the joint-log probability and consequently
    // subtract the derivative of this quantity w.r.t. the weights, we can
    // simply add the derivative (i.e. no need to negate, and we do gradient ascent).
    const size_t num_instances = result.batch_size_;
    const size_t num_ids = 2 * num_instances;

    // Keep track of multipliers for every instance.
    std::unique_ptr<device_matrix<FloatT>> instance_multipliers(
        result.weights_->copy(result.weights_->getStream()));

    CHECK_DIMENSIONS(*instance_multipliers,
                     1 /* num_rows */, num_instances /* num_cols */);

    const cudaStream_t instance_multipliers_stream = merge_streams(
        instance_multipliers->getStream(),
        result.similarity_probs_->getStream());

    CHECK_DIMENSIONS_EQUAL(
        *instance_multipliers,
        *result.similarity_probs_);

    // Multiply learning rate divided by batch size with
    // the per-instance weights.
    const FloatT batch_size_normalizer = exp(-log(result.batch_size_));

    // Multiply with the complement of the probabilities.
    thrust::transform(
        thrust::cuda::par.on(instance_multipliers_stream),
        // Multiply in instance weights.
        begin(*instance_multipliers), // PLACEHOLDER FOR instance weights.
        end(*instance_multipliers), /* first op*/
        // Multiply in batch normalization constant.
        make_scalar_multiplication_iterator(
            // Multiply in similarity probabilities derivatives.
            thrust::make_transform_iterator(
                begin(*result.similarity_probs_),
                func::sigmoid_to_log_sigmoid_deriv<FloatT>(
                    model_->desc_.clip_sigmoid() ? 1e-6 : 0.0 /* epsilon */)),
            batch_size_normalizer), /* second op */
        begin(*instance_multipliers), /* result */
        thrust::multiplies<FloatT>());

    // Get pointers to intermediate results that we will use.
    const device_matrix<FloatT>* const reprs = result.representations_.get();
    CHECK_EQ(reprs->getCols(), num_ids);

    CHECK_DIMENSIONS(*result.representations_,
                     get_representation_storage()->size(), num_ids);

    // d cost / d reprs
    //
    // TODO(cvangysel): remove this const_cast.
    device_matrix<FloatT>* const grad_reprs =
        const_cast<ForwardResultType*>(&result)->representations_.release();

    reset_grad(gradients.get(), grad_reprs);

    // Flip adjacent columns. The matrix is organized in pairs of columns.
    // The gradient w.r.t. the first representation is the representation
    // of the second, and vice versa!
    flip_adjacent_columns(grad_reprs->getStream(), grad_reprs);

    std::unique_ptr<device_matrix<FloatT>> broadcasted_instance_multipliers(
        broadcast_columns(instance_multipliers->getStream(),
                          *instance_multipliers,
                          2 /* num_repeats */));

    // Verify dimensions.
    CHECK_EQ(grad_reprs->getCols(),
             broadcasted_instance_multipliers->getCols());

    // Merge in multipliers.
    apply_columnwise<thrust::multiplies<FloatT>>(
        thrust::cuda::par.on(stream),
        *broadcasted_instance_multipliers,
        grad_reprs);

    return gradients.release();
}

Representations<::Typedefs::FloatT, ::Typedefs::IdxType>* Objective::get_representation_storage() const {
    switch (param_id_) {
    case WORD_REPRS:
        return &model_->words_;
    case ENTITY_REPRS:
        return &model_->entities_;
    default:
        break;
    };

    LOG(FATAL) << "Unable to return representations.";
    throw 0;
}

void Objective::reset_grad(GradientsType* const gradients, device_matrix<FloatT>* const grad_reprs) const {
    switch (param_id_) {
    case WORD_REPRS:
        gradients->grad_phrase_reprs_.reset(grad_reprs);
        return;
    case ENTITY_REPRS:
        gradients->grad_entity_repr_.reset(grad_reprs);
        return;
    default:
        break;
    };

    LOG(FATAL) << "Unable to set gradient.";
    throw 0;
}

}  // namespace RepresentationSimilarity

namespace TextEntityEntityEntity {

Objective::Objective(
        ::Typedefs::ModelBase* const model,
        const lse::TrainConfig& train_config)
    : ::Objective<BatchType, ForwardResultType, GradientsType>(model, train_config),
      text_entity_weight_(train_config.text_entity_weight()),
      entity_entity_weight_(train_config.entity_entity_weight()),
      text_entity_objective_(model, train_config),
      entity_entity_objective_(ENTITY_REPRS, model, train_config) {
    CHECK_NE(text_entity_weight_, 0.0);
    CHECK_NE(entity_entity_weight_, 0.0);
}

Objective::ForwardResultType* Objective::compute_cost(
        const BatchType& batch, RNG* const rng) const {
    PROFILE_FUNCTION();

    return new ForwardResultType(
        std::make_tuple(
            std::make_pair(text_entity_objective_.compute_cost(std::get<0>(batch), rng),
                           text_entity_weight_),
            std::make_pair(entity_entity_objective_.compute_cost(std::get<1>(batch), rng),
                           entity_entity_weight_)));
}

Objective::GradientsType* Objective::compute_gradients(const ForwardResultType& result) {
    PROFILE_FUNCTION();

    MergeGradientsFn<FloatT> merge_gradients_fn;

    std::unique_ptr<Gradients<FloatT>> text_entity_gradients(
        text_entity_objective_.compute_gradients(*std::get<0>(std::get<0>(result.forward_results_))));
    FloatT text_entity_weight = std::get<1>(std::get<0>(result.forward_results_));

    std::unique_ptr<Gradients<FloatT>> entity_entity_gradients(
        entity_entity_objective_.compute_gradients(*std::get<0>(std::get<1>(result.forward_results_))));
    FloatT entity_entity_weight = std::get<1>(std::get<1>(result.forward_results_));

    Objective::GradientsType* const merged_gradients = merge_gradients_fn({
        {text_entity_gradients.release(), text_entity_weight},
        {entity_entity_gradients.release(), entity_entity_weight},
    });

    return merged_gradients;
}

}  // namespace TextEntityEntityEntity

namespace TextEntityTermTerm {

Objective::Objective(
        ::Typedefs::ModelBase* const model,
        const lse::TrainConfig& train_config)
    : ::Objective<BatchType, ForwardResultType, GradientsType>(model, train_config),
      text_entity_weight_(train_config.text_entity_weight()),
      term_term_weight_(train_config.term_term_weight()),
      text_entity_objective_(model, train_config),
      term_term_objective_(WORD_REPRS, model, train_config) {
    CHECK_NE(text_entity_weight_, 0.0);
    CHECK_NE(term_term_weight_, 0.0);
}

Objective::ForwardResultType* Objective::compute_cost(
        const BatchType& batch, RNG* const rng) const {
    PROFILE_FUNCTION();

    return new ForwardResultType(
        std::make_tuple(
            std::make_pair(text_entity_objective_.compute_cost(std::get<0>(batch), rng),
                           text_entity_weight_),
            std::make_pair(term_term_objective_.compute_cost(std::get<1>(batch), rng),
                           term_term_weight_)));
}

Objective::GradientsType* Objective::compute_gradients(const ForwardResultType& result) {
    PROFILE_FUNCTION();

    MergeGradientsFn<FloatT> merge_gradients_fn;

    std::unique_ptr<Gradients<FloatT>> text_entity_gradients(
        text_entity_objective_.compute_gradients(*std::get<0>(std::get<0>(result.forward_results_))));
    FloatT text_entity_weight = std::get<1>(std::get<0>(result.forward_results_));

    std::unique_ptr<Gradients<FloatT>> term_term_gradients(
        term_term_objective_.compute_gradients(*std::get<0>(std::get<1>(result.forward_results_))));
    FloatT term_term_weight = std::get<1>(std::get<1>(result.forward_results_));

    Objective::GradientsType* const merged_gradients = merge_gradients_fn({
        {text_entity_gradients.release(), text_entity_weight},
        {term_term_gradients.release(), term_term_weight},
    });

    return merged_gradients;
}

}  // namespace TextEntityTermTerm
