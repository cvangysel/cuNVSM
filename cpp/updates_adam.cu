#include "cuNVSM/updates.h"

template <typename FloatT>
AdamTransformGradientUpdater<FloatT>::AdamTransformGradientUpdater(
    const size_t source_vector_dim,
    const size_t target_vector_dim,
    Streams* const streams,
    const FloatT beta1,
    const FloatT beta2,
    const FloatT epsilon)
        : TransformGradientUpdater<FloatT>(
            epsilon,
            {new TransformStorage<FloatT>(source_vector_dim,
                                          target_vector_dim,
                                          streams), /* m_prev */
             new TransformStorage<FloatT>(source_vector_dim,
                                          target_vector_dim,
                                          streams) /* v_prev */}),
          beta1_(beta1), beta2_(beta2), t_(1) {}

template <typename FloatT>
void adam_update(const device_matrix<FloatT>& m,
                 const device_matrix<FloatT>& v,
                 device_matrix<FloatT>* grad,
                 const FloatT bias_correction,
                 const FloatT epsilon) {
    CHECK_DIMENSIONS_EQUAL(m, *grad);
    CHECK_DIMENSIONS_EQUAL(v, *grad);

    thrust::transform(
        thrust::make_transform_iterator(
            m.begin(),
            func::scale_by_constant<FloatT>(bias_correction)),
        thrust::make_transform_iterator(
            m.end(),
            func::scale_by_constant<FloatT>(bias_correction)),
        thrust::make_transform_iterator(
            thrust::make_transform_iterator(
                v.begin(),
                func::sqrt<FloatT>()),
            func::add_constant<FloatT>(epsilon)),
        grad->begin(),
        thrust::divides<FloatT>());
}

template <typename FloatT>
void AdamTransformGradientUpdater<FloatT>::update(
        TransformStorage<FloatT>* const storage,
        typename TransformStorage<FloatT>::GradientType* const gradient_desc,
        const FloatT learning_rate,
        const FloatT scaled_regularization_lambda,
        Streams* const streams) {
    CHECK(storage != nullptr);

    apply_regularization(
        streams->next(),
        scaled_regularization_lambda,
        storage,
        gradient_desc);

    // Update m_t.
    dynamic_cast<TransformStorage<FloatT>*>(this->storages_[0].get())->update(
        *gradient_desc,
        1.0 - beta1_, /* learning_rate */
        1.0, /* regularization_lambda */
        streams,
        func::identity<FloatT>() /* update_transform_op */);

    // Update v_t.
    dynamic_cast<TransformStorage<FloatT>*>(this->storages_[1].get())->update(
        *gradient_desc,
        1.0 - beta2_, /* learning_rate */
        1.0, /* regularization_lambda */
        streams,
        func::square<FloatT>() /* update_transform_op */);

    device_matrix<FloatT>& grad_transform = std::get<0>(*gradient_desc);
    device_matrix<FloatT>& grad_bias = std::get<1>(*gradient_desc);

    device_matrix<FloatT>* m_transform;
    device_matrix<FloatT>* m_bias;

    device_matrix<FloatT>* v_transform;
    device_matrix<FloatT>* v_bias;

    std::tie(m_transform, m_bias) =
        dynamic_cast<TransformStorage<FloatT>*>(this->storages_[0].get())->get();
    std::tie(v_transform, v_bias) =
        dynamic_cast<TransformStorage<FloatT>*>(this->storages_[1].get())->get();

    const FloatT bias_correction = sqrt(1.0 - pow(beta2_, t_)) / (1.0 - pow(beta1_, t_));

    adam_update(*m_transform, *v_transform, &grad_transform, bias_correction, this->epsilon_);
    adam_update(*m_bias, *v_bias, &grad_bias, bias_correction, this->epsilon_);

    t_ += 1;

    CHECK_MATRIX(grad_transform);
    CHECK_MATRIX(grad_bias);

    return storage->update(
        *gradient_desc, learning_rate,
        static_cast<FloatT>(0.0) /* regularization_lambda */,
        streams);
}

#define DENSE_UPDATE_DENSE_VARIANCE AdamConf::DENSE_UPDATE_DENSE_VARIANCE
#define DENSE_UPDATE AdamConf::DENSE_UPDATE
#define SPARSE AdamConf::SPARSE

template <typename FloatT, typename IdxType>
AdamRepresentationsGradientUpdater<FloatT, IdxType>::AdamRepresentationsGradientUpdater(
    const size_t num_objects,
    const size_t repr_size,
    const AdamConf& conf,
    Streams* const streams,
    const FloatT beta1,
    const FloatT beta2,
    const FloatT epsilon)
        : RepresentationsGradientUpdater<FloatT, IdxType>(
            epsilon,
            {new RepresentationsStorage<FloatT, IdxType>(num_objects,
                                                         repr_size,
                                                         streams), /* m_prev */
             new RepresentationsStorage<FloatT, IdxType>(num_objects,
                                                         conf.mode() < DENSE_UPDATE_DENSE_VARIANCE ? 1 : repr_size /* repr_size */,
                                                         streams) /* v_prev */}),
          conf_(conf),
          beta1_(beta1), beta2_(beta2),
          t_(1) {}

template <typename FloatT, typename IdxType>
__global__
void adam_sparse_update_kernel(const size_t window_size,
                               const IdxType* const indices,
                               const FloatT* const m,
                               const FloatT* const v,
                               FloatT* const grad,
                               const FloatT bias_correction,
                               const FloatT epsilon) {
    FloatT agg_m = 0.0;
    FloatT agg_v = 0.0;
    for (IdxType w = 0; w < window_size; ++w) {
        agg_m += m[indices[blockIdx.x * window_size + w] * blockDim.x + threadIdx.x];
        agg_v += v[indices[blockIdx.x * window_size + w]];
    }
    agg_m /= window_size;
    agg_v /= window_size;

    grad[blockIdx.x * blockDim.x + threadIdx.x] = bias_correction * agg_m / (::sqrt(agg_v) + epsilon);
}

template <typename FloatT, typename IdxType>
void AdamRepresentationsGradientUpdater<FloatT, IdxType>::update(
        RepresentationsStorage<FloatT, IdxType>* const storage,
        typename RepresentationsStorage<FloatT, IdxType>::GradientType* const gradient_descs,
        const FloatT learning_rate,
        const FloatT scaled_regularization_lambda,
        Streams* const streams) {
    CHECK(storage != nullptr);

    const bool use_sgd_regularization = (conf_.mode() < DENSE_UPDATE_DENSE_VARIANCE);

    if (use_sgd_regularization) {
        LOG_IF_EVERY_N(WARNING, scaled_regularization_lambda > 0.0, 10000)
            << "Sparse variants of Adam currently do not correctly implement l2 regularization.";
    }

    RepresentationsStorage<FloatT, IdxType>* const m_storage =
        dynamic_cast<RepresentationsStorage<FloatT, IdxType>*>(
            this->storages_[0].get());
    RepresentationsStorage<FloatT, IdxType>* const v_storage =
        dynamic_cast<RepresentationsStorage<FloatT, IdxType>*>(
            this->storages_[1].get());

    device_matrix<FloatT>* m = m_storage->get();
    device_matrix<FloatT>* v = v_storage->get();

    // Invariants.
    for (const typename RepresentationsStorage<FloatT, IdxType>::SingleGradientType& gradient_desc : *gradient_descs) {
        device_matrix<FloatT>& gradient = std::get<0>(gradient_desc);
        const size_t repr_size = gradient.getRows();
        const size_t num_grads = gradient.getCols();

        CHECK_EQ((dynamic_cast<RepresentationsStorage<FloatT, IdxType>*>(
                      this->storages_[0].get())->get()->getRows()),
                 repr_size);

        const device_matrix<IdxType>& indices = std::get<1>(gradient_desc);
        const size_t window_size = std::get<2>(gradient_desc);

        CHECK_EQ(indices.getCols() % window_size, 0);
    }

    // Update m_t.
    dynamic_cast<RepresentationsStorage<FloatT, IdxType>*>(this->storages_[0].get())->update(
        *gradient_descs,
        1.0 - beta1_, /* learning_rate */
        1.0, /* regularization_lambda */
        streams);

    // Add regularization within m_t.
    if (!use_sgd_regularization) {
        // m_t = beta1 * m_{t-1} + (1.0 - beta1) grad
        //    with grad = grad - lambda * params
        // m_t = beta1 * m_{t-1} + (1.0 - beta1) grad - (1.0 - beta1) * lambda * params

        apply_regularization(
            streams->next(),
            static_cast<FloatT>((1.0 - beta1_) * scaled_regularization_lambda),
            storage->get(),
            m);
    }

    // Update v_t.
    if (conf_.mode() < DENSE_UPDATE_DENSE_VARIANCE) {
        std::vector<std::unique_ptr<device_matrix<FloatT>>> matrix_ptrs;  // For memory management.
        std::vector<typename RepresentationsStorage<FloatT, IdxType>::SingleGradientType> average_squared_gradients;

        for (const typename RepresentationsStorage<FloatT, IdxType>::SingleGradientType& gradient_desc : *gradient_descs) {
            device_matrix<FloatT>& gradient = std::get<0>(gradient_desc);

            matrix_ptrs.push_back(
                std::unique_ptr<device_matrix<FloatT>>(
                    new device_matrix<FloatT>(
                        1, /* num_rows */
                        gradient.getCols(),
                        gradient.getStream())));

            device_matrix<FloatT>* const average_squared_gradient = matrix_ptrs.back().get();

            reduce_axis<FloatT, func::square<FloatT>>(
                average_squared_gradient->getStream(),
                FIRST_AXIS,
                gradient,
                average_squared_gradient);

            average_squared_gradient->scale(
                average_squared_gradient->getStream(),
                exp(-log(gradient.getRows())));

            average_squared_gradients.push_back(
                std::forward_as_tuple(*average_squared_gradient,
                                      std::get<1>(gradient_desc),
                                      std::get<2>(gradient_desc),
                                      std::get<3>(gradient_desc)));
        }

        v_storage->update(average_squared_gradients,
                          1.0 - beta2_, /* learning_rate */
                          1.0, /* regularization_lambda */
                          streams);
    } else {
        CHECK(!use_sgd_regularization);

        std::unique_ptr<RepresentationsStorage<FloatT, IdxType>> agg_repr_grad(
            new RepresentationsStorage<FloatT, IdxType>(
                v_storage->num_objects(),
                v_storage->repr_size(),
                DefaultStream::get()));

        agg_repr_grad->initialize_with_null();

        agg_repr_grad->update(*gradient_descs,
                              1.0, /* learning_rate */
                              0.0, /* scaled_regularization_lambda */
                              streams);

        apply_regularization(
            streams->next(),
            scaled_regularization_lambda,
            storage->get(),
            agg_repr_grad->get());

        agg_repr_grad->get()->square(agg_repr_grad->get()->getStream()); // g_t^2 

        v_storage->update_dense(merge_streams(v->getStream(),
                                              agg_repr_grad->get()->getStream()),
                                begin(*agg_repr_grad->get()),
                                1.0 - beta2_, /* learning_rate */
                                1.0 /* regularization_lambda */);
    }

    // Compute update.
    const FloatT bias_correction = sqrt(1.0 - pow(beta2_, t_)) / (1.0 - pow(beta1_, t_));

    t_ += 1;

    if (conf_.mode() >= DENSE_UPDATE) {
        const cudaStream_t m_v_stream = merge_streams(
            m->getStream(), v->getStream());

        if (conf_.mode() == DENSE_UPDATE) {
            return storage->update_dense(
                m_v_stream,
                thrust::make_transform_iterator(
                    thrust::make_transform_iterator(
                        thrust::make_zip_iterator(
                            thrust::make_tuple(
                                begin(*m),
                                thrust::make_transform_iterator(
                                    thrust::make_transform_iterator(
                                        thrust::make_permutation_iterator(
                                            begin(*v), /* elements */
                                            make_matrix_column_iterator(*m) /* map */),
                                        func::sqrt<FloatT>()),
                                    func::add_constant<FloatT>(this->epsilon_)))),
                        func::divides_tuple<FloatT>()),
                    func::scale_by_constant<FloatT>(bias_correction)),
                learning_rate,
                scaled_regularization_lambda);
        } else if (conf_.mode() == DENSE_UPDATE_DENSE_VARIANCE) {
            return storage->update_dense(
                    m_v_stream,
                    thrust::make_transform_iterator(
                        thrust::make_transform_iterator(
                            thrust::make_zip_iterator(
                                thrust::make_tuple(
                                    begin(*m),
                                    thrust::make_transform_iterator(
                                        thrust::make_transform_iterator(
                                            begin(*v), /* elements */
                                            func::sqrt<FloatT>()),
                                        func::add_constant<FloatT>(this->epsilon_)))),
                            func::divides_tuple<FloatT>()),
                        func::scale_by_constant<FloatT>(bias_correction)),
                    learning_rate,
                    0.0 /* scaled_regularzation_lambda */);
        } else {
            LOG(FATAL) << "Invalid mode configuration.";
        }
    } else {
        // This is a variant of the sparse implementation of Adam.
        //
        // Statistics are kept on a per-representation level, but updates are averaged over
        // all objects in one window.
        //
        // The true sparse algorithm would track statistics per representation, but not
        // spread the updates over all objects in one window. It would simply load the
        // right update for every object in the batch. However, this would require
        // deduplicating (i.e., sorting) the object indices, resizing the indices and update
        // tensors and then copying. This is probably more expensive than simply computing the full
        // update, as this avoids the deduplication step (albeit more expensive memory-wise,
        // depending on the batch size, corpus size and the distribution of instances).
        //
        // TODO(cvangysel): implement an option to compute the full gradient.

        CHECK_EQ(gradient_descs->size(), 1)
            << "Sparse Adam currently does not implement multiple gradients.";

        const typename RepresentationsStorage<FloatT, IdxType>::SingleGradientType& gradient_desc = gradient_descs->front();

        device_matrix<FloatT>& gradient = std::get<0>(gradient_desc);
        const size_t repr_size = gradient.getRows();
        const size_t num_grads = gradient.getCols();

        const device_matrix<IdxType>& indices = std::get<1>(gradient_desc);
        const size_t window_size = std::get<2>(gradient_desc);

        LAUNCH_KERNEL(
            adam_sparse_update_kernel<<<num_grads, /* num_blocks */
                                        repr_size, /* threads_per_block */
                                        0,
                                        merge_streams(
                                           m->getStream(),
                                           v->getStream())>>>(
                window_size,
                indices.getData(),
                m->getData(),
                v->getData(),
                gradient.getData(),
                bias_correction,
                this->epsilon_));

        CHECK_MATRIX(gradient);

        return storage->update(
            *gradient_descs,
            learning_rate,
            use_sgd_regularization
                ? scaled_regularization_lambda
                : static_cast<FloatT>(0.0), /* scaled_regularization_lambda */
            streams);
    }
}

// Explicit instantiations.
template class AdamTransformGradientUpdater<FLOATING_POINT_TYPE>;
template class AdamRepresentationsGradientUpdater<FLOATING_POINT_TYPE, int32>;