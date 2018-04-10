#ifndef CUNVSM_STORAGE_INL_H
#define CUNVSM_STORAGE_INL_H

template <typename FloatT,
          typename GradientIterator,
          typename UpdateTransformOp = thrust::identity<FloatT>,
          typename AggOp = thrust::plus<FloatT>>
void update_dense(const cudaStream_t stream,
                  device_matrix<FloatT>* const param,
                  GradientIterator grad_param_it,
                  const FloatT learning_rate,
                  const FloatT scaled_regularization_lambda,
                  const UpdateTransformOp update_transform_op = UpdateTransformOp(),
                  const AggOp agg_op = AggOp()) {
    // param + grad_param; element-wise addition.
    thrust::transform(
        thrust::cuda::par.on(stream),
        make_scalar_multiplication_iterator(
            begin(*param),
            1.0 - scaled_regularization_lambda * learning_rate), /* first op */
        make_scalar_multiplication_iterator(
            end(*param),
            1.0 - scaled_regularization_lambda * learning_rate),
        make_scalar_multiplication_iterator(
            thrust::make_transform_iterator(
                grad_param_it, update_transform_op),
            learning_rate), /* second op */
        begin(*param), /* dest */
        agg_op);

    CHECK_MATRIX(*param);
}

template <typename FloatT,
          typename UpdateTransformOp = thrust::identity<FloatT>,
          typename AggOp = thrust::plus<FloatT>>
void update_dense(device_matrix<FloatT>* const param,
                  const device_matrix<FloatT>& grad_param,
                  const FloatT learning_rate,
                  const FloatT scaled_regularization_lambda,
                  const UpdateTransformOp update_transform_op = UpdateTransformOp(),
                  const AggOp agg_op = AggOp()) {
    update_dense(
        merge_streams(param->getStream(), grad_param.getStream()),
        param,
        begin(grad_param),
        learning_rate,
        scaled_regularization_lambda,
        update_transform_op,
        agg_op);
}

#endif /* CUNVSM_STORAGE_INL_H */