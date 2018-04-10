#include "cuNVSM/updates.h"

template <typename FloatT>
GradientUpdater<FloatT>::GradientUpdater(
    const FloatT epsilon,
    const std::vector<Storage<FloatT>*>& storages)
        : epsilon_(epsilon), storages_() {
    CHECK_GT(epsilon_, 0.0);

    for (Storage<FloatT>* const storage : storages) {
        // Make everything null.
        storage->initialize_with_null();

        // Add to container.
        storages_.push_back(std::unique_ptr<Storage<FloatT>>(storage));
    }
}

// Explicit instantiations.
template class GradientUpdater<FLOATING_POINT_TYPE>;

// SGD.

template <typename FloatT>
void SGDTransformGradientUpdater<FloatT>::update(
        TransformStorage<FloatT>* const storage,
        typename TransformStorage<FloatT>::GradientType* const gradient_desc,
        const FloatT learning_rate,
        const FloatT scaled_regularization_lambda,
        Streams* const streams) {
    CHECK(storage != nullptr);

    return storage->update(
        *gradient_desc, learning_rate, scaled_regularization_lambda, streams);
}

template <typename FloatT, typename IdxType>
void SGDRepresentationsGradientUpdater<FloatT, IdxType>::update(
        RepresentationsStorage<FloatT, IdxType>* const storage,
        typename RepresentationsStorage<FloatT, IdxType>::GradientType* const gradient_desc,
        const FloatT learning_rate,
        const FloatT scaled_regularization_lambda,
        Streams* const streams) {
    CHECK(storage != nullptr);

    return storage->update(
        *gradient_desc, learning_rate, scaled_regularization_lambda, streams);
}

// Explicit instantiations.
template class SGDTransformGradientUpdater<FLOATING_POINT_TYPE>;
template class SGDRepresentationsGradientUpdater<FLOATING_POINT_TYPE, int32>;