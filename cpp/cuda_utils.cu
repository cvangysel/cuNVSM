#include "cuNVSM/cuda_utils.h"

template <typename FloatT>
Normalizer<FloatT>::Normalizer(const size_t num_instances)
        : num_instances_(num_instances),
          input_cache_(nullptr),
          norms_(new device_matrix<FloatT>(
              1, num_instances_, NULL /* stream */)) {
    CHECK_GE(num_instances_, 1);
}

template <typename FloatT>
void Normalizer<FloatT>::forward(
        const device_matrix<FloatT>& input,
        device_matrix<FloatT>* const output) {
    CHECK_EQ(input.getCols(), num_instances_);
    CHECK_DIMENSIONS_EQUAL(input, *output);

    if (input_cache_ == nullptr || !input_cache_->hasSameShape(input)) {
        input_cache_.reset(input.copy(input.getStream()));
    } else {
        input_cache_->copyFrom(input_cache_->getStream(), input);
    }

    reduce_axis(merge_streams(input.getStream(), norms_->getStream()),
                FIRST_AXIS,
                input,
                norms_.get(),
                func::square<FloatT>());

    apply_elemwise<func::sqrt<FloatT>>(
        thrust::cuda::par.on(norms_->getStream()),
        norms_.get());

    apply_columnwise<thrust::divides<FloatT>,
                     thrust::identity<FloatT>,
                     thrust::identity<FloatT>>(
        thrust::cuda::par.on(
            merge_streams(
                merge_streams(input.getStream(),
                              norms_->getStream()),
                output->getStream())),
        input,
        *norms_,
        output);
}

template <typename FloatT>
device_matrix<FloatT>* hadamard_product_and_reduce(
        const device_matrix<FloatT>& first_op,
        const device_matrix<FloatT>& second_op) {
    CHECK_DIMENSIONS_EQUAL(first_op, second_op);

    std::unique_ptr<device_matrix<FloatT>> multiplied(
        hadamard_product(
            merge_streams(first_op.getStream(), second_op.getStream()),
            first_op, second_op));

    std::unique_ptr<device_matrix<FloatT>> reduced(
        new device_matrix<FloatT>(1, first_op.getCols(), multiplied->getStream()));

    reduce_axis(reduced->getStream(),
                FIRST_AXIS,
                *multiplied,
                reduced.get());

    return reduced.release();
}

template <typename FloatT>
void Normalizer<FloatT>::backward(
        const device_matrix<FloatT>& grad_output,
        device_matrix<FloatT>* const grad_input) {
    CHECK(input_cache_ != nullptr);
    CHECK(norms_ != nullptr);
    CHECK(input_cache_->hasSameShape(grad_output))
        << *input_cache_ << " vs. " << grad_output;

    CHECK_NE(&grad_output, grad_input);

    apply_columnwise<thrust::multiplies<FloatT>,
                     thrust::identity<FloatT>, /* first_op_op */
                     func::square<FloatT> /* second_op_op */>(
        thrust::cuda::par.on(
            merge_streams(
                merge_streams(grad_output.getStream(),
                              norms_->getStream()),
                grad_input->getStream())),
        grad_output,
        *norms_,
        grad_input);

    std::unique_ptr<device_matrix<FloatT>> grad_output_input_cross(
        hadamard_product_and_reduce(*input_cache_, grad_output));

    apply_columnwise<thrust::multiplies<FloatT>,
                     thrust::identity<FloatT>, /* first_op_op */
                     thrust::identity<FloatT> /* second_op_op */>(
        thrust::cuda::par.on(
            merge_streams(input_cache_->getStream(),
                          grad_output_input_cross->getStream())),
        *input_cache_,
        *grad_output_input_cross,
        input_cache_.get());

    elemwise_plus(
        thrust::cuda::par.on(
            merge_streams(input_cache_->getStream(),
                          grad_input->getStream())),
        *input_cache_,
        grad_input,
        func::scale_by_constant<FloatT>(-1.0));

    apply_columnwise<thrust::divides<FloatT>>(
        thrust::cuda::par.on(
            merge_streams(
                merge_streams(grad_output.getStream(),
                              norms_->getStream()),
                grad_input->getStream())),
        *grad_input,
        *norms_,
        grad_input,
        thrust::identity<FloatT>(), /* first_op_op */
        func::power<FloatT>(3.0) /* second_op_op */);

    CHECK_MATRIX(*grad_input);

    // Resets state.
    input_cache_.reset();
}

template <typename FloatT>
device_matrix<FloatT>* Normalizer<FloatT>::backward(
        const device_matrix<FloatT>& grad_output) {
    device_matrix<FloatT>* grad_input = device_matrix<FloatT>::create_shape_as(
        grad_output.getStream(), grad_output);

    this->backward(grad_output, grad_input);

    return grad_input;
}

// Explicit instantiation.
template class Normalizer<FLOATING_POINT_TYPE>;
