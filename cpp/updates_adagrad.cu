#include "cuNVSM/updates.h"

template <typename FloatT>
AdagradTransformGradientUpdater<FloatT>::AdagradTransformGradientUpdater(
    const size_t source_vector_dim,
    const size_t target_vector_dim,
    Streams* const streams,
    const FloatT epsilon)
        : TransformGradientUpdater<FloatT>(
            epsilon,
            {new TransformStorage<FloatT>(source_vector_dim,
                                          target_vector_dim,
                                          streams)}) {}

template <typename FloatT>
void adagrad_update(const device_matrix<FloatT>& agg_grad,
                    device_matrix<FloatT>* grad,
                    const FloatT epsilon) {
    CHECK_DIMENSIONS_EQUAL(agg_grad, *grad);

    thrust::transform(
        grad->begin(),
        grad->end(),
        thrust::make_transform_iterator(
            thrust::make_transform_iterator(
                agg_grad.begin(),
                func::add_constant<FloatT>(epsilon)),
            func::sqrt<FloatT>()),
        grad->begin(),
        thrust::divides<FloatT>());
}

template <typename FloatT>
void AdagradTransformGradientUpdater<FloatT>::update(
        TransformStorage<FloatT>* const storage,
        typename TransformStorage<FloatT>::GradientType* const gradient_desc,
        const FloatT learning_rate,
        const FloatT scaled_regularization_lambda,
        Streams* const streams) {
    CHECK(storage != nullptr);

    LOG_IF_EVERY_N(WARNING, scaled_regularization_lambda > 0.0, 10000)
        << "Adagrad currently does not correctly implement l2 regularization.";

    // Update accumulated gradient.
    dynamic_cast<TransformStorage<FloatT>*>(this->storages_[0].get())->update(
        *gradient_desc,
        1.0, /* learning_rate */
        0.0, /* regularization_lambda */
        streams,
        func::square<FloatT>());

    device_matrix<FloatT>& grad_transform = std::get<0>(*gradient_desc);
    device_matrix<FloatT>& grad_bias = std::get<1>(*gradient_desc);

    device_matrix<FloatT>* agg_grad_transform;
    device_matrix<FloatT>* agg_grad_bias;

    std::tie(agg_grad_transform, agg_grad_bias) =
        dynamic_cast<TransformStorage<FloatT>*>(this->storages_[0].get())->get();

    adagrad_update(*agg_grad_transform, &grad_transform, this->epsilon_);
    adagrad_update(*agg_grad_bias, &grad_bias, this->epsilon_);

    CHECK_MATRIX(grad_transform);
    CHECK_MATRIX(grad_bias);

    return storage->update(
        *gradient_desc, learning_rate, scaled_regularization_lambda, streams);
}

template <typename FloatT, typename IdxType>
AdagradRepresentationsGradientUpdater<FloatT, IdxType>::AdagradRepresentationsGradientUpdater(
    const size_t num_objects,
    Streams* const streams,
    const FloatT epsilon)
        : RepresentationsGradientUpdater<FloatT, IdxType>(
            epsilon,
            {new RepresentationsStorage<FloatT, IdxType>(num_objects,
                                                         1 /* repr_size */,
                                                         streams)}) {}

template <typename FloatT, typename IdxType>
__global__
void adagrad_update_kernel(const size_t window_size,
                           const IdxType* const indices,
                           const FloatT* const agg_grad,
                           FloatT* const grad,
                           const FloatT epsilon) {
    FloatT agg_agg_grad = 0.0;
    for (IdxType w = 0; w < window_size; ++w) {
        agg_agg_grad += agg_grad[indices[blockIdx.x * window_size + w]];
    }
    agg_agg_grad /= window_size;

    grad[blockIdx.x * blockDim.x + threadIdx.x] /= ::sqrt(agg_agg_grad + epsilon);
}

template <typename FloatT, typename IdxType>
void AdagradRepresentationsGradientUpdater<FloatT, IdxType>::update(
        RepresentationsStorage<FloatT, IdxType>* const storage,
        typename RepresentationsStorage<FloatT, IdxType>::GradientType* const gradient_descs,
        const FloatT learning_rate,
        const FloatT scaled_regularization_lambda,
        Streams* const streams) {
    CHECK(storage != nullptr);

    CHECK_EQ(gradient_descs->size(), 1)
        << "Adagrad currently does not implement multiple gradients.";

    typename RepresentationsStorage<FloatT, IdxType>::SingleGradientType* const gradient_desc = &gradient_descs->front();

    LOG_IF_EVERY_N(WARNING, scaled_regularization_lambda > 0.0, 15000)
        << "Adagrad currently does not correctly implement l2 regularization.";

    device_matrix<FloatT>& gradient = std::get<0>(*gradient_desc);
    const size_t num_grads = gradient.getCols();
    const size_t repr_size = gradient.getRows();

    const device_matrix<IdxType>& indices = std::get<1>(*gradient_desc);
    const size_t window_size = std::get<2>(*gradient_desc);

    const device_matrix<FloatT>* const repr_weights = std::get<3>(*gradient_desc);

    // CHECK(repr_weights == (device_matrix<FloatT>*) nullptr)
    //     << "Adagrad currently does not correctly implement weighted representations.";
    LOG_IF_EVERY_N(WARNING, repr_weights == (device_matrix<FloatT>*) nullptr, 15000)
        << "Adagrad currently does not correctly implement weighted representations.";

    CHECK_EQ(indices.getCols() % window_size, 0);

    device_matrix<FloatT> average_gradient(1, /* num_rows */
                                           gradient.getCols(),
                                           gradient.getStream());

    reduce_axis<FloatT, func::square<FloatT>>(
        average_gradient.getStream(),
        FIRST_AXIS,
        gradient,
        &average_gradient);

    average_gradient.scale(average_gradient.getStream(),
                           exp(-log(gradient.getRows())));

    typename RepresentationsStorage<FloatT, IdxType>::SingleGradientType average_gradient_desc =
        std::forward_as_tuple(
            average_gradient,
            std::get<1>(*gradient_desc),
            std::get<2>(*gradient_desc),
            std::get<3>(*gradient_desc) /* TODO(cvangysel): verify this. */);

    // Update accumulated gradient.
    dynamic_cast<RepresentationsStorage<FloatT, IdxType>*>(this->storages_[0].get())->update(
        {average_gradient_desc},
        1.0, /* learning_rate */
        0.0, /* regularization_lambda */
        streams,
        func::identity<FloatT>());

    device_matrix<FloatT>* agg_grad_reprs =
        dynamic_cast<RepresentationsStorage<FloatT, IdxType>*>(this->storages_[0].get())->get();

    // TODO(cvangysel): verify this when repr_weights is not NULL.
    LAUNCH_KERNEL(
        adagrad_update_kernel<<<num_grads, /* num_blocks */
                                repr_size, /* threads_per_block */
                                0,
                                agg_grad_reprs->getStream()>>>(
            window_size,
            indices.getData(),
            agg_grad_reprs->getData(),
            gradient.getData(),
            this->epsilon_));

    CHECK_MATRIX(gradient);

    return storage->update(
        *gradient_descs, learning_rate, scaled_regularization_lambda, streams);
}

// Explicit instantiations.
template class AdagradTransformGradientUpdater<FLOATING_POINT_TYPE>;
template class AdagradRepresentationsGradientUpdater<FLOATING_POINT_TYPE, int32>;