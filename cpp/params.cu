#include "cuNVSM/base.h"
#include "cuNVSM/cuda_utils.h"
#include "cuNVSM/params.h"

#include <map>
#include <set>

#include <glog/logging.h>

class ConstructGradient {
 public:
  template <typename FloatT>
  static typename TransformStorage<FloatT>::GradientType* construct(
        const typename Storage<FloatT>::Gradients& gradients,
        const TransformStorage<FloatT>& param,
        const ParamIdentifier param_id) {
    return gradients.get_transform_gradient(param_id);
  }

  template <typename FloatT, typename IdxType>
  static typename RepresentationsStorage<FloatT, IdxType>::GradientType* construct(
        const typename Storage<FloatT>::Gradients& gradients,
        const RepresentationsStorage<FloatT, IdxType>& param,
        const ParamIdentifier param_id) {
    return gradients.get_representations_gradient(param_id);
  }
};

char const* ParamName[] {
    "word_representations",
    "word_entity_mapping",
    "entity_representations",
};

template <typename FloatT, typename IdxType>
Representations<FloatT, IdxType>::Representations(
        const ParamIdentifier id,
        const size_t num_objects,
        const size_t size,
        const UpdateMethodConf& update_method,
        Streams* const streams)
        : Parameters<FloatT>(id),
          RepresentationsStorage<FloatT, IdxType>(num_objects, size, streams),
          updater_(nullptr) {
    if (update_method.type() == SGD) {
        updater_.reset(
            new SGDRepresentationsGradientUpdater<FloatT, IdxType>());
    } else if (update_method.type() == ADAGRAD) {
        updater_.reset(
            new AdagradRepresentationsGradientUpdater<FloatT, IdxType>(
                num_objects, streams));
    } else if (update_method.type() == ADAM) {
        updater_.reset(
            new AdamRepresentationsGradientUpdater<FloatT, IdxType>(
                num_objects, size,
                update_method.adam_conf(), /* dense */
                streams));

    }

    CHECK(updater_ != nullptr);
}

template <typename FloatT, typename IdxType>
Representations<FloatT, IdxType>::~Representations() {}

template <typename FloatT, typename IdxType>
void Representations<FloatT, IdxType>::initialize(RNG* const rng) {
    PROFILE_FUNCTION();

    init_matrix_glorot(reprs_.getStream(), &reprs_, rng);
    Parameters<FloatT>::initialize(rng);
}

template <typename FloatT, typename IdxType>
__global__
void average_repr_kernel(const size_t window_size,
                         const FloatT* const repr,
                         const IdxType* const indices,
                         const FloatT* const indices_weights,
                         FloatT* const avg_repr) {
    const IdxType indices_idx = blockIdx.x * window_size;

    FloatT agg = 0.0;
    for (IdxType w = 0; w < window_size; ++w) {
        const IdxType repr_idx = indices[indices_idx + w];
        const FloatT repr_weight =
            indices_weights == nullptr ?
            1.0 : indices_weights[indices_idx + w];

        agg += repr_weight * repr[blockDim.x * repr_idx + threadIdx.x];
    }

    avg_repr[blockIdx.x * blockDim.x + threadIdx.x] = agg / window_size;
}

template <typename FloatT, typename IdxType>
device_matrix<FloatT>* Representations<FloatT, IdxType>::get_representations(
        const cudaStream_t stream,
        const device_matrix<IdxType>& indices) const {
    PROFILE_FUNCTION_WITH_STREAM(stream);

    const size_t num_requested = indices.size();
    DCHECK_GT(num_requested, 0);

    device_matrix<FloatT>* const requested_reprs = new device_matrix<FloatT>(
        size(), num_requested, stream);

    LAUNCH_KERNEL(
        average_repr_kernel<<<num_requested, /* num_blocks */
                              size(), /* threads_per_block */
                              0,
                              stream>>>(
            static_cast<size_t>(1) /* window_size */,
            reprs_.getData(), /* repr_input */
            indices.getData(), /* idx_input */
            (FloatT*) nullptr, /* idx_input_weights */
            requested_reprs->getData() /* output */));

    return requested_reprs;
}

template <typename FloatT, typename IdxType>
device_matrix<FloatT>* Representations<FloatT, IdxType>::get_representation(const IdxType idx) const {
    PROFILE_FUNCTION();

    const cudaStream_t stream = DefaultStream::get()->next();

    std::unique_ptr<device_matrix<FloatT>> repr(
        new device_matrix<FloatT>(size(), 1, stream));

    thrust::copy(reprs_.begin(idx), reprs_.begin(idx + 1),
                 repr->begin());

    return repr.release();
}

template <typename FloatT, typename IdxType>
device_matrix<FloatT>* Representations<FloatT, IdxType>::get_average_representations(
        const cudaStream_t stream,
        const device_matrix<IdxType>& indices,
        const size_t window_size,
        const device_matrix<FloatT>* const indices_weights) const {
    PROFILE_FUNCTION_WITH_STREAM(stream);

    const int32 num_requested = indices.size();
    const int32 num_average_requested = num_requested / window_size;

    if (indices_weights != nullptr) {
        CHECK_EQ(num_requested, indices_weights->size());
    }

    DCHECK_EQ(num_requested % window_size, 0);
    DCHECK_GT(num_average_requested, 0);

    device_matrix<FloatT>* avg_requested_reps =
        new device_matrix<FloatT>(size(), num_average_requested,
                                  stream);

    LAUNCH_KERNEL(
        average_repr_kernel<<<num_average_requested, /* num_blocks */
                              size(), /* threads_per_block */
                              0,
                              stream>>>(
            window_size,
            reprs_.getData(), /* repr_input */
            indices.getData(), /* idx_input */
            indices_weights != nullptr ? indices_weights->getData() : (FloatT*) nullptr, /* idx_input_weights */
            avg_requested_reps->getData() /* output */));

    return avg_requested_reps;
}

template <typename FloatT, typename IdxType>
device_matrix<FloatT>* Representations<FloatT, IdxType>::compute_similarity(
        const device_matrix<FloatT>& first,
        const device_matrix<FloatT>& second) const {
    PROFILE_FUNCTION();

    CHECK_DIMENSIONS_EQUAL(first, second);

    const cudaStream_t stream = merge_streams(
        first.getStream(), second.getStream());

    std::unique_ptr<device_matrix<FloatT>> multiplied_reprs(
        hadamard_product(stream, first, second));

    device_matrix<FloatT>* const similarities = new device_matrix<FloatT>(
        1, /* num_rows */
        multiplied_reprs->getCols(),
        multiplied_reprs->getStream());

    reduce_axis(
        similarities->getStream(),
        FIRST_AXIS,
        *multiplied_reprs,
        similarities);

    return similarities;
}

template <typename FloatT, typename IdxType>
std::vector<std::vector<FloatT>> Representations<FloatT, IdxType>::compute_similarity(
        const device_matrix<FloatT>& query_input_vectors,
        const std::vector<IdxType>& indices) const {
    PROFILE_FUNCTION();

    CHECK_EQ(query_input_vectors.getRows(), size());
    const size_t num_query_vectors = query_input_vectors.getCols();
    const size_t num_candidate_representations = indices.size();

    std::unique_ptr<device_matrix<IdxType>> d_indices(
        device_matrix<IdxType>::create_column(query_input_vectors.getStream(), indices));

    // get_representations ignores dimensionality of d_indices.
    std::unique_ptr<device_matrix<FloatT>> candidate_representations(
        get_representations(reprs_.getStream(), *d_indices));

    // Broadcast the candidate representation vectors, such that [e1 e2 e3]
    // becomes [e1 e2 e3 e1 e2 e3] if num_query_vectors == 2.
    std::unique_ptr<device_matrix<FloatT>> repeated_candidate_representations(
        repmat(candidate_representations->getStream(),
               *candidate_representations,
               num_query_vectors /* num_repeats */));

    // Holds the repeated query vectors and afterwards the multiplied representations.
    //
    // Broadcasts the query vectors such that [q1 q2 q3] becomes [q1 q1 q2 q2 q3 q3] if
    // if num_candidate_representations == 2.
    std::unique_ptr<device_matrix<FloatT>> broadcasted_query_vectors(
        broadcast_columns(query_input_vectors.getStream(),
                          query_input_vectors,
                          num_candidate_representations));

    CHECK_DIMENSIONS_EQUAL(*repeated_candidate_representations,
                           *broadcasted_query_vectors);

    std::unique_ptr<device_matrix<FloatT>> d_similarities(
        compute_similarity(*repeated_candidate_representations,
                           *broadcasted_query_vectors));

    const FloatT* const flattened_similarities = get_array(
        d_similarities->getStream(), *d_similarities);

    std::vector<std::vector<FloatT>> similarities;

    for (size_t i = 0;
         i < num_candidate_representations * num_query_vectors;
         i += num_candidate_representations) {
        similarities.push_back(std::vector<FloatT>(
            flattened_similarities + i,
            flattened_similarities + i + num_candidate_representations));
    }

    delete [] flattened_similarities;

    return similarities;
}

template <typename FloatT, typename IdxType>
FloatT Representations<FloatT, IdxType>::compute_similarity(const IdxType first,
                                                            const IdxType second) const {
    CHECK_LT(first, num_objects());
    CHECK_LT(second, num_objects());

    std::unique_ptr<device_matrix<IdxType>> first_indices(
        device_matrix<IdxType>::create_column(reprs_.getStream(),
                                              {first}));
    std::unique_ptr<device_matrix<IdxType>> second_indices(
        device_matrix<IdxType>::create_column(reprs_.getStream(),
                                              {second}));

    // get_representations ignores dimensionality of d_indices.
    std::unique_ptr<device_matrix<FloatT>> first_representations(
        get_representations(reprs_.getStream(), *first_indices));
    inplace_l2_normalize_columns(first_representations.get());

    std::unique_ptr<device_matrix<FloatT>> second_representations(
        get_representations(reprs_.getStream(), *second_indices));
    inplace_l2_normalize_columns(second_representations.get());

    std::unique_ptr<device_matrix<FloatT>> similarities(
        compute_similarity(*first_representations, *second_representations));

    const FloatT* const flattened_similarities = get_array(
        similarities->getStream(), *similarities);

    const FloatT similarity = flattened_similarities[0];

    delete [] flattened_similarities;

    return similarity;
}

template <typename FloatT, typename IdxType>
void Representations<FloatT, IdxType>::update(
        const typename Storage<FloatT>::Gradients& gradients,
        const FloatT learning_rate,
        const FloatT scaled_regularization_lambda,
        Streams* const streams) {
    typename std::unique_ptr<typename RepresentationsStorage<FloatT, IdxType>::GradientType> gradient_desc(
        ConstructGradient::construct(gradients, *this, this->id_));

    // No gradient.
    if (gradient_desc.get() == nullptr) {
        return;
    }

    updater_->update(this, gradient_desc.get(),
                     learning_rate, scaled_regularization_lambda,
                     streams);
}

template <typename FloatT, typename IdxType>
FloatT Representations<FloatT, IdxType>::get_parameter_gradient(
        const typename Storage<FloatT>::Gradients& gradients,
        const size_t idx) const {
    std::unique_ptr<const typename RepresentationsStorage<FloatT, IdxType>::GradientType> gradient_desc(
        ConstructGradient::construct(gradients, *this, this->id_));

    if (gradient_desc == nullptr) {
        return 0.0;
    } else {
        return RepresentationsStorage<FloatT, IdxType>::get_parameter_gradient(
            *gradient_desc, idx);
    }
}

template <typename FloatT>
Transform<FloatT>::Transform(
        const ParamIdentifier id,
        const lse::ModelDesc::TransformDesc& desc,
        const size_t word_repr_size,
        const size_t entity_repr_size,
        const UpdateMethodConf& update_method,
        Streams* const streams)
        : Parameters<FloatT>(id),
          TransformStorage<FloatT>(word_repr_size, entity_repr_size, streams),
          desc_(desc),
          updater_(nullptr) {
    if (update_method.type() == SGD) {
        updater_.reset(
            new SGDTransformGradientUpdater<FloatT>());
    } else if (update_method.type() == ADAGRAD) {
        updater_.reset(
            new AdagradTransformGradientUpdater<FloatT>(
                source_repr_size(),
                target_repr_size(),
                streams));
    } else if (update_method.type() == ADAM) {
        updater_.reset(
            new AdamTransformGradientUpdater<FloatT>(
                source_repr_size(),
                target_repr_size(),
                streams));
    }

    CHECK(updater_ != nullptr);
}

template <typename FloatT>
void Transform<FloatT>::initialize(RNG* const rng) {
    PROFILE_FUNCTION();

    // Randomly initialize word-to-entity mapping.
    init_matrix_glorot(transform_.getStream(), &transform_, rng);

    // Set bias to null.
    bias_.fillwith(bias_.getStream(), 0.0);

    Parameters<FloatT>::initialize(rng);
}

template <typename FloatT>
Transform<FloatT>::~Transform() {}

template <typename FloatT>
device_matrix<FloatT>* Transform<FloatT>::transform(
        const cudaStream_t stream,
        const device_matrix<FloatT>& word_repr,
        BatchNormalization<FloatT>* const batch_normalization) const {
    PROFILE_FUNCTION_WITH_STREAM(stream);

    DCHECK_EQ(word_repr.getRows(), source_repr_size());
    DCHECK_GE(word_repr.getCols(), 1);

    CHECK_MATRIX(word_repr);

    const size_t num_instances = word_repr.getCols();

    CHECK_MATRIX(bias_);

    // Variable 'res' will hold the result, as well as the bias vector.
    device_matrix<FloatT>* transformed = nullptr;

    if (batch_normalization != nullptr) {
        transformed = new device_matrix<FloatT>(
            target_repr_size(), num_instances,
            merge_streams(stream, bias_.getStream()));
    } else {
        transformed = broadcast_columns(
            merge_streams(stream, bias_.getStream()),
            bias_, /* src */
            num_instances /* num_repeats */);
    }

    CHECK_DIMENSIONS((*transformed),
                     target_repr_size(), num_instances);

    CHECK_MATRIX(*transformed);

    const cudaStream_t params_steam = merge_streams(
        stream, transform_.getStream());

    // transform_ is entity_repr_size by word_repr_size
    // word_repr is word_repr_size by num_words
    matrix_mult(stream,
                transform_, CUBLAS_OP_N,
                word_repr, CUBLAS_OP_N,
                transformed, /* dst */
                (batch_normalization == nullptr) /* dst_contains_bias */);

    CHECK_MATRIX(*transformed);

    if (batch_normalization != nullptr) {
        batch_normalization->forward(
            *transformed, bias_, transformed);
    }

    switch (desc_.nonlinearity()) {
        case lse::ModelDesc::TransformDesc::TANH:
            apply_elemwise<func::tanh<FloatT>>(
                thrust::cuda::par.on(stream),
                transformed);

            break;
        case lse::ModelDesc::TransformDesc::HARD_TANH:
            apply_elemwise<func::clip<FloatT>>(
                thrust::cuda::par.on(stream),
                transformed,
                func::clip<FloatT>(-1.0, 1.0));

            break;
        default:
            LOG(FATAL) << "nonlinearity " << desc_.nonlinearity() << " not implemented.";
    };

    CHECK_MATRIX(*transformed);

    return transformed;
}

template <typename FloatT>
void Transform<FloatT>::backward(
        const cudaStream_t stream,
        const typename Storage<FloatT>::ForwardResult& result,
        const device_matrix<FloatT>& broadcasted_input,
        const device_matrix<FloatT>& output,
        device_matrix<FloatT>* const grad_output,
        Gradients<FloatT>* const gradients) const {
    std::unique_ptr<typename TransformStorage<FloatT>::GradientType> gradient_desc_ptr(
        ConstructGradient::construct(*gradients, *this, this->id_));

    CHECK_NOTNULL(gradient_desc_ptr.get());

    typename TransformStorage<FloatT>::GradientType gradient_desc = *gradient_desc_ptr;

    CHECK_DIMENSIONS_EQUAL(*grad_output, output);

    device_matrix<FloatT>* const grad_transform_matrix = &std::get<0>(gradient_desc);
    device_matrix<FloatT>* const grad_bias = &std::get<1>(gradient_desc);

    // d cost / d (Wx + b)
    switch (desc_.nonlinearity()) {
        case lse::ModelDesc::TransformDesc::TANH:
            hadamard_product(
                thrust::cuda::par.on(stream),
                output, /* first operand */
                grad_output, /* second operand and destination */
                func::tanh_to_sech2<FloatT>() /* operation for first operand */);

            break;
        case lse::ModelDesc::TransformDesc::HARD_TANH:
            hadamard_product(
                thrust::cuda::par.on(stream),
                output, /* first operand */
                grad_output, /* second operand and destination */
                func::clip_to_clip_deriv<FloatT>(-1.0, 1.0) /* operation for first operand */);

            break;
    };

    // d cost / d bias; reduce_axis does not expect nulled output.
    MAKE_MATRIX_NULL(*grad_bias);

    // TODO(cvangysel): get rid of this horrible construct.
    const TextEntity::ForwardResult<
        FloatT,
        typename Storage<FloatT>::ForwardResult::WordIdxType,
        typename Storage<FloatT>::ForwardResult::EntityIdxType>* textentity_result =
        dynamic_cast<
            const TextEntity::ForwardResult<
                FloatT,
                typename Storage<FloatT>::ForwardResult::WordIdxType,
                typename Storage<FloatT>::ForwardResult::EntityIdxType>*>(&result);

    CHECK_NOTNULL(textentity_result);

    if (textentity_result->batch_normalization_ == nullptr) {
        reduce_axis(
            stream,
            SECOND_AXIS,
            *grad_output,
            grad_bias);
    } else {
        textentity_result->batch_normalization_->backward(
            *grad_output,
            bias_,
            grad_output,
            grad_bias);
    }

    // d cost / d transform_matrix
    //
    // TODO(cvangysel): figure out whether we actually need the copy here?!
    MAKE_MATRIX_NULL(*grad_transform_matrix); // TODO(cvangysel): figure out whether we need this.

    matrix_mult(stream,
                *grad_output, CUBLAS_OP_N,
                broadcasted_input, CUBLAS_OP_T,
                grad_transform_matrix);

    CHECK_MATRIX(*grad_transform_matrix);
    CHECK_MATRIX(*grad_bias);
}

template <typename FloatT>
void Transform<FloatT>::update(
        const typename Storage<FloatT>::Gradients& gradients,
        const FloatT learning_rate,
        const FloatT scaled_regularization_lambda,
        Streams* const streams) {
    std::unique_ptr<typename TransformStorage<FloatT>::GradientType> gradient_desc(
        ConstructGradient::construct(gradients, *this, this->id_));

    // No gradient.
    if (gradient_desc == nullptr) {
        return;
    }

    updater_->update(this, gradient_desc.get(),
                     learning_rate, scaled_regularization_lambda,
                     streams);
}

template <typename FloatT>
FloatT Transform<FloatT>::get_parameter_gradient(
        const typename Storage<FloatT>::Gradients& gradients,
        const size_t idx) const {
    std::unique_ptr<const typename TransformStorage<FloatT>::GradientType> gradient_desc(
        ConstructGradient::construct(gradients, *this, this->id_));

    if (gradient_desc.get() == nullptr) {
        return 0.0;
    } else {
        return TransformStorage<FloatT>::get_parameter_gradient(*gradient_desc, idx);
    }
}

// Explicit instantiations.
template class Parameters<FLOATING_POINT_TYPE>;
template class Representations<FLOATING_POINT_TYPE, int32>;
template class Transform<FLOATING_POINT_TYPE>;
