#include "cuNVSM/storage.h"

#include <thrust/iterator/transform_iterator.h>

template <typename FloatT, typename IdxType>
RepresentationsStorage<FloatT, IdxType>::RepresentationsStorage(
        const size_t num_objects,
        const size_t size,
        Streams* const streams)
    : reprs_(size, num_objects, streams->next()) {
    PROFILE_FUNCTION();

    DCHECK_GT(num_objects, 0);
    DCHECK_GT(size, 0);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
// From http://stackoverflow.com/questions/16077464/atomicadd-for-double-on-gpu.
//
// This is a hack that allows the tests to run in double precision.
// atomicAdd for doubles is available in CUDA 8 and onwards.
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// Optimizations for loading the gradient into shared memory inspired by
// https://devblogs.nvidia.com/parallelforall/how-access-global-memory-efficiently-cuda-c-kernels/
template <typename FloatT, typename IdxType>
__global__
void update_repr_kernel(const FloatT learning_rate,
                        const FloatT* const gradient,
                        const IdxType* const indices,
                        FloatT* const repr,
                        const FloatT* const repr_weights) {
    const FloatT weight = repr_weights != nullptr ? repr_weights[blockIdx.x * gridDim.y + blockIdx.y] : 1.0;

    // Update.
    atomicAdd(&repr[indices[blockIdx.x * gridDim.y + blockIdx.y] * blockDim.x + threadIdx.x],
              learning_rate * weight * gradient[blockIdx.x * blockDim.x + threadIdx.x]);
}

template <typename FloatT, typename IdxType>
template <typename UpdateTransformOp, typename AggOp>
void RepresentationsStorage<FloatT, IdxType>::update(
        const GradientType& gradient_descs,
        const FloatT learning_rate,
        const FloatT scaled_regularization_lambda,
        Streams* const streams,
        const UpdateTransformOp update_transform_op,
        const AggOp agg_op) {
    PROFILE_FUNCTION();

    CHECK_GE(learning_rate, 0.0);
    CHECK_GE(scaled_regularization_lambda, 0.0);

    if (scaled_regularization_lambda > 0.0) {
        reprs_.scale(reprs_.getStream(), 1.0 - (scaled_regularization_lambda * learning_rate));
    }

    for (const SingleGradientType& gradient_desc : gradient_descs) {
        const device_matrix<FloatT>& repr_grad = std::get<0>(gradient_desc);
        const device_matrix<IdxType>& indices = std::get<1>(gradient_desc);
        const size_t window_size = std::get<2>(gradient_desc);
        const device_matrix<FloatT>* const repr_weights = std::get<3>(gradient_desc);

        if (repr_weights != nullptr) {
            CHECK_DIMENSIONS(*repr_weights, 1, indices.getCols());
        }

        CHECK_EQ(indices.getRows(), 1);

        // Indices should be a multiple of window_size.
        CHECK_EQ(indices.size() % window_size, 0);

        CHECK_DIMENSIONS(repr_grad,
                         repr_size(), indices.size() / window_size);

        const size_t num_grads = repr_grad.getCols();

        LAUNCH_KERNEL(
            update_repr_kernel<<<dim3(num_grads, window_size), /* num_blocks */
                                 repr_size(), /* threads_per_block */
                                 0,
                                 streams->next() /* stream */>>>(
                learning_rate,
                repr_grad.getData(), /* repr_grad */
                indices.getData(), /* idx_input */
                reprs_.getData(), /* output */
                repr_weights != nullptr ? repr_weights->getData() : nullptr));
    }

    CHECK_MATRIX(reprs_);
}

template <typename FloatT, typename IdxType>
device_matrix<FloatT>* RepresentationsStorage<FloatT, IdxType>::get() {
    return &reprs_;
}

template <typename FloatT, typename IdxType>
void RepresentationsStorage<FloatT, IdxType>::initialize_with_constant(const FloatT value) {
    reprs_.fillwith(reprs_.getStream(), value);
}

template <typename FloatT, typename IdxType>
typename Storage<FloatT>::DataType RepresentationsStorage<FloatT, IdxType>::get_data() const {
    PROFILE_FUNCTION();

    return {
        {"representations", &reprs_},
    };
}

template <typename FloatT, typename IdxType>
void RepresentationsStorage<FloatT, IdxType>::increment_parameter(
        const size_t idx, const FloatT epsilon) {
    PROFILE_FUNCTION();

    DCHECK_LT(idx, num_parameters());

    increment_scalar(epsilon, raw_begin(reprs_) + idx);
}

template <typename FloatT, typename IdxType>
FloatT RepresentationsStorage<FloatT, IdxType>::get_parameter_gradient(
        const GradientType& gradient_descs,
        const size_t param_idx) const {
    PROFILE_FUNCTION();

    DCHECK_GE(param_idx, 0);
    DCHECK_LT(param_idx, num_parameters());

    FloatT grad_param = 0.0;

    for (const SingleGradientType& gradient_desc : gradient_descs) {
        const device_matrix<FloatT>& repr_grad = std::get<0>(gradient_desc);

        const device_matrix<IdxType>& d_indices = std::get<1>(gradient_desc);
        const thrust::host_vector<IdxType> indices(begin(d_indices), end(d_indices));

        const size_t window_size = std::get<2>(gradient_desc);

        const device_matrix<FloatT>* const repr_weights = std::get<3>(gradient_desc);

        CHECK_DIMENSIONS(repr_grad,
                         repr_size(), indices.size() / window_size);
        CHECK_EQ(indices.size() % window_size, 0);

        // Compute the representation identifier.
        const size_t repr_idx = param_idx / repr_size();
        const size_t repr_param_idx = param_idx % repr_size();

        size_t g_idx = 0;
        for (const IdxType idx : indices) {
            if (repr_idx == idx) {
                const size_t grad_offset = (g_idx / window_size) * repr_size();
                const FloatT* const scalar_ptr = raw_begin(repr_grad) + grad_offset + repr_param_idx;

                const FloatT grad_param_weight_ = (
                    repr_weights != nullptr
                    ? get_scalar(raw_begin(*repr_weights) + g_idx) : 1.0);

                const FloatT grad_param_ = grad_param_weight_*  get_scalar(scalar_ptr);
                grad_param += grad_param_;
            }

            ++g_idx;
        }

        CHECK_EQ(g_idx, repr_grad.getCols() * window_size);
    }

    return grad_param;
}

template <typename FloatT>
TransformStorage<FloatT>::TransformStorage(
        const size_t word_repr_size,
        const size_t entity_repr_size,
        Streams* const streams)
        : transform_(entity_repr_size, word_repr_size, streams->next()),
          bias_(entity_repr_size, 1, streams->next()) {
    PROFILE_FUNCTION();

    CHECK_GT(word_repr_size, 0);
    CHECK_GT(entity_repr_size, 0);
}

template <typename FloatT>
template <typename UpdateTransformOp, typename AggOp>
void TransformStorage<FloatT>::update(
        const GradientType& gradient_desc,
        const FloatT learning_rate,
        const FloatT scaled_regularization_lambda,
        Streams* const streams,
        const UpdateTransformOp update_transform_op,
        const AggOp agg_op) {
    PROFILE_FUNCTION();

    const device_matrix<FloatT>& grad_matrix = std::get<0>(gradient_desc);
    const device_matrix<FloatT>& grad_bias = std::get<1>(gradient_desc);

    CHECK_GE(learning_rate, 0.0);
    CHECK_GE(scaled_regularization_lambda, 0.0);

    // W + grad_W; element-wise addition.
    update_dense(&transform_,
                 grad_matrix,
                 learning_rate,
                 scaled_regularization_lambda,
                 update_transform_op, agg_op);

    // b + grad_b; element-wise addition.
    update_dense(&bias_,
                 grad_bias,
                 learning_rate,
                 static_cast<FloatT>(0.0), /* bias should not be regularized */
                 update_transform_op, agg_op);
}

template <typename FloatT>
std::tuple<device_matrix<FloatT>*,
           device_matrix<FloatT>*> TransformStorage<FloatT>::get() {
    return std::make_tuple(&transform_, &bias_);
}

template <typename FloatT>
void TransformStorage<FloatT>::initialize_with_constant(const FloatT value) {
    transform_.fillwith(transform_.getStream(), value);
    bias_.fillwith(bias_.getStream(), value);
}

template <typename FloatT>
typename Storage<FloatT>::DataType TransformStorage<FloatT>::get_data() const {
    PROFILE_FUNCTION();

    return {
        {"transform", &transform_},
        {"bias", &bias_},
    };
}

template <typename FloatT>
void TransformStorage<FloatT>::increment_parameter(
        const size_t idx, const FloatT epsilon) {
    PROFILE_FUNCTION();

    DCHECK_GE(idx, 0);
    DCHECK_LT(idx, num_parameters());

    FloatT* const scalar_ptr = (idx < transform_.size()) ?
        raw_begin(transform_) + idx :
        raw_begin(bias_) + (idx - transform_.size());

    increment_scalar(epsilon, scalar_ptr);
}

template <typename FloatT>
FloatT TransformStorage<FloatT>::get_parameter_gradient(
        const GradientType& gradient_desc,
        const size_t idx) const {
    PROFILE_FUNCTION();

    const device_matrix<FloatT>& grad_transform = std::get<0>(gradient_desc);
    const device_matrix<FloatT>& grad_bias = std::get<1>(gradient_desc);

    CHECK_DIMENSIONS_EQUAL(grad_transform, transform_);
    CHECK_DIMENSIONS_EQUAL(grad_bias, bias_);

    DCHECK_GE(idx, 0);
    DCHECK_LT(idx, num_parameters());

    const FloatT* const scalar_ptr = (idx < grad_transform.size()) ?
        raw_begin(grad_transform) + idx :
        raw_begin(grad_bias) + (idx - grad_transform.size());

    return get_scalar(scalar_ptr);
}

// Explicit instantiations.

template class Storage<FLOATING_POINT_TYPE>;

template class TransformStorage<FLOATING_POINT_TYPE>;
#define INSTANTIATE_TransformStorage(FloatT, UpdateTransformOp, AggOp) \
template void TransformStorage<FloatT>::update<UpdateTransformOp<FloatT>, AggOp<FloatT>>( \
    const TransformStorage<FloatT>::GradientType&, \
    const FloatT, \
    const FloatT, \
    Streams* const, \
    const UpdateTransformOp<FloatT>, \
    const AggOp<FloatT>)

INSTANTIATE_TransformStorage(FLOATING_POINT_TYPE, func::identity, thrust::plus);
INSTANTIATE_TransformStorage(FLOATING_POINT_TYPE, func::square, thrust::plus);
#undef INSTANTIATE_TransformStorage

template class RepresentationsStorage<FLOATING_POINT_TYPE, int32>;
#define INSTANTIATE_RepresentationsStorage(FloatT, IdxType, UpdateTransformOp, AggOp) \
template void RepresentationsStorage<FloatT, IdxType>::update<UpdateTransformOp<FloatT>, AggOp<FloatT>>( \
    const GradientType&, \
    const FloatT, \
    const FloatT, \
    Streams* const, \
    const UpdateTransformOp<FloatT>, \
    const AggOp<FloatT>)

INSTANTIATE_RepresentationsStorage(FLOATING_POINT_TYPE, int32, func::identity, thrust::plus);
// INSTANTIATE_RepresentationsStorage(FLOATING_POINT_TYPE, int32, func::square, func::identity, thrust::plus);

#undef INSTANTIATE_RepresentationsStorage
