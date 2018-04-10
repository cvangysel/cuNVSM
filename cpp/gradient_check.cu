#include "cuNVSM/gradient_check.h"

template <typename ModelT>
bool GradientCheckFn<ModelT>::operator()(
        ModelT* const model,
        const typename ModelT::Batch& batch,
        const typename ModelT::ForwardResult& result,
        const typename ModelT::Gradients& gradients,
        const FloatT epsilon,
        const FloatT relative_error_threshold,
        const std::stringstream& rng_state,
        RNG* const rng) {
    PROFILE_FUNCTION();
    DCHECK(model->initialized());

    CHECK_GE(epsilon, 0.0);
    CHECK_GE(relative_error_threshold, 0.0);

    DCHECK(!rng_state.eof() && rng_state.good());

    // Disable verbose logging;
    const int32 verbose_loglevel = FLAGS_v;
    FLAGS_v = 1;

    const cudaStream_t stream = DefaultStream::get()->next();

    // Sanity check to make sure we're getting the right RNG state.
    CHECK_EQ(result.get_cost(),
             model->get_cost(batch, &rng_state, rng));

    bool checked = true;

    for (const auto& pair : model->params_) {
        const ParamIdentifier param_id = pair.first;
        Parameters<FloatT>* const param = pair.second;

        Storage<FloatT>* const storage =
            dynamic_cast<Storage<FloatT>*>(pair.second);
        CHECK_NOTNULL(storage);

        for (size_t param_idx = 0; param_idx < storage->num_parameters(); ++param_idx) {
            const FloatT gradient_predict =
                - param->get_parameter_gradient(gradients, param_idx); // TODO(cvangysel): remove negation

            // Add epsilon to weight.
            storage->increment_parameter(param_idx, epsilon);

            // Compute cost with epsilon added to weight.
            const FloatT cost_added_epsilon = model->get_cost(
                batch, &rng_state, rng);

            // Subtract epsilon from weight.
            storage->increment_parameter(param_idx, -2.0 * epsilon);

            // Compute cost with epsilon removed from weight.
            const FloatT cost_removed_epsilon = model->get_cost(
                batch, &rng_state, rng);

            // Restore original weight.
            storage->increment_parameter(param_idx, epsilon);

            const FloatT gradient_approx =
                (cost_added_epsilon - cost_removed_epsilon) /
                (2.0 * epsilon);

            const FloatT relative_error =
                abs(gradient_predict - gradient_approx) /
                max(abs(gradient_predict), abs(gradient_approx));

            const FloatT ratio = (gradient_approx != 0.0) ?
                (gradient_predict / gradient_approx) : NAN;

            if (gradient_predict * gradient_approx < 0.0) {
                LOG(ERROR) << "Parameter " << param_idx << " of "
                           << ParamName[param_id] << " has gradient with incorrect direction "
                           << "(approx=" << gradient_approx << ", "
                           << "predict=" << gradient_predict << ", "
                           << "ratio=" << ratio << ", "
                           << "relative error=" << relative_error << ").";

                checked = false;
            } else if (relative_error >= relative_error_threshold) {
                VLOG(1) << "Parameter " << param_idx << " of "
                        << ParamName[param_id] << " most likely has incorrect gradient "
                        << "(approx=" << gradient_approx << ", "
                        << "predict=" << gradient_predict << ", "
                        << "ratio=" << ratio << ", "
                        << "relative error=" << relative_error << ").";

                if (!std::isnan(ratio)) {
                    checked = false;
                }
            } else if ((gradient_approx != 0.0) || (gradient_predict != 0.0)) {
                VLOG(2) << "Parameter " << param_idx << " of "
                        << ParamName[param_id] << " has correct gradient "
                        << "(approx=" << gradient_approx << ", "
                        << "predict=" << gradient_predict << ", "
                        << "ratio=" << ratio << ", "
                        << "relative error=" << relative_error << ").";
            }

            // TODO(cvangysel): CHECK_DOUBLE_EQ was failing here, while the gradient was checking.
            //                  This happened after upgrading to CUDA 8; it's probably better to check for the relative error here.
            CHECK_NEAR(
                result.get_cost(),
                model->get_cost(batch, &rng_state, rng),
                1e-5);
        }
    }

    // TODO(cvangysel): same comment as above.
    // Sanity check to make sure we're getting the right RNG state.
    CHECK_NEAR(
        result.get_cost(),
        model->get_cost(batch, &rng_state, rng),
        1e-5);

    // Enable verbose logging again.
    FLAGS_v = verbose_loglevel;

    if (!checked) {
        CHECK_NOTNULL(result.get_similarity_probs());

        FloatT* data = get_array(result.get_similarity_probs()->getStream(),
                                 *result.get_similarity_probs());
        std::vector<FloatT> probs(data, data + result.get_similarity_probs()->size());
        delete [] data;

        LOG(ERROR) << "Similarity probs: " << probs;
    }

    return checked;
}

// Explicit instantiations.
template class GradientCheckFn<LSE>;
template class GradientCheckFn<Model<EntityEntity::Objective>>;
template class GradientCheckFn<Model<TermTerm::Objective>>;
template class GradientCheckFn<Model<TextEntityEntityEntity::Objective>>;
template class GradientCheckFn<Model<TextEntityTermTerm::Objective>>;
