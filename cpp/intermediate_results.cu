#include "cuNVSM/intermediate_results.h"

template <typename FloatT>
Gradients<FloatT>* MergeGradientsFn<FloatT>::operator()(const GradientAndWeights& gradients_and_weights) const {
    CHECK_GT(gradients_and_weights.size(), 1);

    std::vector<Gradients<FloatT>*> gradients;
    gradients.reserve(gradients_and_weights.size());

    for (auto it = gradients_and_weights.begin();
         it != gradients_and_weights.end();
         ++it) {
        Gradients<FloatT>* const grad = std::get<0>(*it);
        gradients.push_back(grad);
    }

    std::unique_ptr<Gradients<FloatT>> result(new CompositeGradients<FloatT>(gradients));

    FloatT summed_weight = 0.0;
    for (auto& pair : gradients_and_weights) {
        summed_weight += std::get<1>(pair);
    }

    for (auto it = gradients_and_weights.begin();
         it != gradients_and_weights.end();
         ++it) {
        Gradients<FloatT>* const grad = std::get<0>(*it);
        const FloatT weight = std::get<1>(*it);

        // Scale the gradients.
        for (auto member_ptr : {&Gradients<FloatT>::grad_entity_repr_,
                                &Gradients<FloatT>::grad_phrase_reprs_,
                                &Gradients<FloatT>::grad_transform_matrix_,
                                &Gradients<FloatT>::grad_bias_}) {
            if ((grad->*member_ptr) != nullptr) {
                (grad->*member_ptr)->scale((grad->*member_ptr)->getStream(), weight / summed_weight);
            }
        }

        // Merge gradients.
        for (auto member_ptr : {&Gradients<FloatT>::grad_transform_matrix_,
                                &Gradients<FloatT>::grad_bias_}) {
            if ((grad->*member_ptr) != nullptr) {
                if ((result.get()->*member_ptr) == nullptr) {
                    (result.get()->*member_ptr).reset((grad->*member_ptr).release());
                } else {
                    elemwise_plus(
                        thrust::cuda::par.on(merge_streams((result.get()->*member_ptr)->getStream(),
                                                           (grad->*member_ptr)->getStream())),
                        *(grad->*member_ptr),
                        (result.get()->*member_ptr).get());

                    (grad->*member_ptr).reset();  // Release memory of constituent.
                }
            }
        }
    }

    return result.release();
}

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
SimpleForwardResult<FloatT, WordIdxType, EntityIdxType>::SimpleForwardResult(
        const size_t batch_size,
        const FloatT regularization_lambda)
    : batch_size_(batch_size),
      regularization_lambda_(regularization_lambda),

      similarity_probs_(nullptr),
      pointwise_mass_(nullptr),

      cost_(NAN) {
    CHECK_GT(batch_size_, 0);
}

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
SimpleForwardResult<FloatT, WordIdxType, EntityIdxType>::SimpleForwardResult()
    : batch_size_(0), regularization_lambda_(0.0), cost_(NAN) {}

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
FloatT SimpleForwardResult<FloatT, WordIdxType, EntityIdxType>::get_cost() const {
    if (std::isnan(cost_) && pointwise_mass_.get() != nullptr) {
        PROFILE_FUNCTION_WITH_STREAM(pointwise_mass_->getStream());

        FloatT log_data_prob;

        {
            device_matrix<FloatT> device_log_data_prob(
                1, 1,
                pointwise_mass_->getStream());

            CHECK_MATRIX(*pointwise_mass_);

            reduce_axis(device_log_data_prob.getStream(),
                        SECOND_AXIS,
                        *pointwise_mass_,
                        &device_log_data_prob);

            FloatT* log_data_prob_ptr;
            CCE(cudaHostAlloc(&log_data_prob_ptr,
                              1 * sizeof(FloatT),
                              cudaHostAllocDefault));

            device_log_data_prob.transfer(
                device_log_data_prob.getStream(),
                log_data_prob_ptr, 1 /* num */);

            cudaStreamSynchronize(device_log_data_prob.getStream());

            log_data_prob = *log_data_prob_ptr;
            cudaFreeHost(log_data_prob_ptr);

            log_data_prob /= this->batch_size_;
        }

        CHECK(isfinite(log_data_prob));

        *const_cast<FloatT*>(&cost_) =
            - log_data_prob +
            this->scaled_regularization_lambda() * 0.0; // TODO(cvangysel): fill in regularization in cost?
    }

    return cost_;
}

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
FloatT SimpleForwardResult<FloatT, WordIdxType, EntityIdxType>::scaled_regularization_lambda() const {
    return this->regularization_lambda_ / this->batch_size_;
}

template <typename FloatT, typename ... ForwardResultT>
class ConstructorFn {
 public:
  ConstructorFn(const std::tuple<std::pair<ForwardResultT*, FloatT> ...>& src,
                std::tuple<std::pair<std::unique_ptr<ForwardResultT>, FloatT> ...>& dst)
      : src_(&src), dst_(&dst) {}

  template <typename Index>
  void operator()(const Index& idx) {
      std::get<0>(std::get<Index::value>(*dst_)).reset(std::get<0>(std::get<Index::value>(*src_)));
      std::get<1>(std::get<Index::value>(*dst_)) = std::get<1>(std::get<Index::value>(*src_));
  }

 private:
  const std::tuple<std::pair<ForwardResultT*, FloatT> ...>* const src_;
  std::tuple<std::pair<std::unique_ptr<ForwardResultT>, FloatT> ...>* const dst_;

  DISALLOW_COPY_AND_ASSIGN(ConstructorFn);
};

template <typename FloatT, typename WordIdxT, typename EntityIdxT, typename ... ForwardResultT>
class HStackFn {
 public:
  typedef device_matrix<FloatT>* (ForwardResult<FloatT, WordIdxT, EntityIdxT>::*MatrixForwardResultFn)() const;

  HStackFn(const std::tuple<std::pair<std::unique_ptr<ForwardResultT>, FloatT> ...>& forward_results,
           MatrixForwardResultFn fn)
      : forward_results_(&forward_results), fn_(fn), args_() {}

  template <typename Index>
  void operator()(const Index& idx) {
      args_.push_back(
        std::make_pair(((std::get<0>(std::get<Index::value>(*forward_results_)).get())->*fn_)(),
                       std::get<1>(std::get<Index::value>(*forward_results_))));
  }

  device_matrix<FloatT>* get_result() const {
      return hstack(DefaultStream::get()->next(), args_);
  }

 private:
  const std::tuple<std::pair<std::unique_ptr<ForwardResultT>, FloatT> ...>* const forward_results_;
  MatrixForwardResultFn fn_;

  std::vector<std::pair<device_matrix<FloatT>*, FloatT>> args_;

  DISALLOW_COPY_AND_ASSIGN(HStackFn);
};

template <typename FloatT, typename WordIdxT, typename EntityIdxT, typename ... ForwardResultT>
MultiForwardResultBase<FloatT, WordIdxT, EntityIdxT, ForwardResultT ...>::MultiForwardResultBase(
        const ForwardResultsType& forward_results)
        : similarity_probs_(nullptr) {
    for_tuple_range(forward_results_, ConstructorFn<FloatT, ForwardResultT ...>(
        forward_results, forward_results_));

    HStackFn<FloatT, WordIdxT, EntityIdxT, ForwardResultT ...> fn(
        this->forward_results_, &ForwardResult<FloatT, WordIdxT, EntityIdxT>::get_similarity_probs);

    for_tuple_range(forward_results_, fn);

    similarity_probs_.reset(fn.get_result());
}

template <typename FloatT, typename WordIdxT, typename EntityIdxT, typename ... ForwardResultT>
class AverageFn {
 public:
  typedef FloatT (ForwardResult<FloatT, WordIdxT, EntityIdxT>::*ScalarForwardResultFn)() const;

  AverageFn(const std::tuple<std::pair<std::unique_ptr<ForwardResultT>, FloatT> ...>& forward_results,
            ScalarForwardResultFn fn)
      : forward_results_(&forward_results), fn_(fn), agg_(0.0) {}

  template <typename Index>
  void operator()(const Index& idx) {
      agg_ += ((std::get<0>(std::get<Index::value>(*forward_results_)).get())->*fn_)();
  }

  FloatT get_result() const {
      return agg_ / std::tuple_size<std::tuple<ForwardResultT ...>>::value;
  }

 private:
  const std::tuple<std::pair<std::unique_ptr<ForwardResultT>, FloatT> ...>* const forward_results_;
  ScalarForwardResultFn fn_;

  FloatT agg_;

  DISALLOW_COPY_AND_ASSIGN(AverageFn);
};

template <typename FloatT, typename WordIdxT, typename EntityIdxT, typename ... ForwardResultT>
FloatT MultiForwardResultBase<FloatT, WordIdxT, EntityIdxT, ForwardResultT ...>::get_cost() const {
    AverageFn<FloatT, WordIdxT, EntityIdxT, ForwardResultT ...> fn(
        forward_results_, &ForwardResult<FloatT, WordIdxT, EntityIdxT>::get_cost);

    for_tuple_range(forward_results_, fn);

    return fn.get_result();
}

template <typename FloatT, typename WordIdxT, typename EntityIdxT, typename ... ForwardResultT>
FloatT MultiForwardResultBase<FloatT, WordIdxT, EntityIdxT, ForwardResultT ...>::scaled_regularization_lambda() const {
    AverageFn<FloatT, WordIdxT, EntityIdxT, ForwardResultT ...> fn(
        forward_results_, &ForwardResult<FloatT, WordIdxT, EntityIdxT>::scaled_regularization_lambda);

    for_tuple_range(forward_results_, fn);

    return fn.get_result();
}

template <typename FloatT, typename WordIdxT, typename EntityIdxT, typename ... ForwardResultT>
device_matrix<FloatT>* MultiForwardResultBase<FloatT, WordIdxT, EntityIdxT, ForwardResultT ...>::get_similarity_probs() const {
    return similarity_probs_.get();
}

template <typename ... Types>
std::tuple<Types&& ...>* forward_as_tuple_ptr(Types&& ... args) {
    return new std::tuple<Types&& ...>(
        std::forward<Types>(args) ...);
}

template <typename FloatT, typename IntT>
std::tuple<device_matrix<FloatT>&, const device_matrix<IntT>&, const size_t>* forward_as_tuple_ptr(
        device_matrix<FloatT>& a, device_matrix<IntT>& b, const size_t c) {
    return new std::tuple<device_matrix<FloatT>&, const device_matrix<IntT>&, const size_t>(a, b, c);
}

template <typename FloatT, typename IntT>
std::tuple<device_matrix<FloatT>&, const device_matrix<IntT>&, const size_t, const device_matrix<FloatT>*>* forward_as_tuple_ptr(
        device_matrix<FloatT>& a, device_matrix<IntT>& b, const size_t c, const device_matrix<FloatT>* d) {
    return new std::tuple<device_matrix<FloatT>&, const device_matrix<IntT>&, const size_t, const device_matrix<FloatT>*>(a, b, c, d);
}

template <typename FloatT>
typename TransformStorage<FloatT>::GradientType* Gradients<FloatT>::get_transform_gradient(
        const ParamIdentifier param_id) const {
    switch (param_id) {
    case TRANSFORM:
        if (this->grad_transform_matrix_ != nullptr) {
            return forward_as_tuple_ptr(*this->grad_transform_matrix_, *this->grad_bias_);
        } else {
            return nullptr;
        }
    default:
    break;
    };

    LOG(FATAL) << "Unable to construct gradient.";
    throw 0;
}

template <typename FloatT>
typename RepresentationsStorage<FloatT, int32>::GradientType* SingleGradients<FloatT>::get_representations_gradient(
        const ParamIdentifier param_id) const {
    switch (param_id) {
    case WORD_REPRS:
        if (this->grad_phrase_reprs_ != nullptr) {
            DCHECK_GT(result_->get_window_size(), 0);

            return new std::vector<typename RepresentationsStorage<FloatT, typename Storage<FloatT>::ForwardResult::WordIdxType>::SingleGradientType> {
                std::forward_as_tuple(
                    *this->grad_phrase_reprs_,
                    *result_->get_word_indices(),
                    result_->get_window_size(),
                    static_cast<const device_matrix<FloatT>*>(result_->get_word_weights()))};
        } else {
            return nullptr;
        }
    case ENTITY_REPRS:
        if (this->grad_entity_repr_ != nullptr) {
            return new std::vector<typename RepresentationsStorage<FloatT, typename Storage<FloatT>::ForwardResult::EntityIdxType>::SingleGradientType> {
                std::forward_as_tuple(
                    *this->grad_entity_repr_,
                    *result_->get_entity_indices(),
                    static_cast<size_t>(1), /* window size */
                    static_cast<const device_matrix<FloatT>*>(nullptr) /* weights */)};
        } else {
            return nullptr;
        }
    default:
        break;
    };

    LOG(FATAL) << "Unable to construct gradient.";
    throw 0;
}

template <typename FloatT>
typename RepresentationsStorage<FloatT, int32>::GradientType* CompositeGradients<FloatT>::get_representations_gradient(
        const ParamIdentifier param_id) const {
    typename RepresentationsStorage<FloatT, int32>::GradientType* gradients =
        new typename RepresentationsStorage<FloatT, int32>::GradientType;

    gradients->reserve(constituent_gradients_.size());

    for (auto& constituent_gradients : constituent_gradients_) {
        std::unique_ptr<
            std::vector<
                typename RepresentationsStorage<FloatT, typename Storage<FloatT>::ForwardResult::WordIdxType>
            ::SingleGradientType>> v(
                constituent_gradients->get_representations_gradient(param_id));

        if (v != nullptr) {
            std::move(v->begin(), v->end(), std::back_inserter(*gradients));
        }
    }

    CHECK_LE(gradients->size(), constituent_gradients_.size());

    return gradients;
}

namespace TextEntity {

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
ForwardResult<FloatT, WordIdxType, EntityIdxType>::ForwardResult(
        device_matrix<WordIdxType>* const flattened_words,
        device_matrix<FloatT>* const flattened_word_weights,
        device_matrix<EntityIdxType>* const entity_ids,
        const size_t window_size,
        const size_t num_random_entities,
        const FloatT regularization_lambda)
    : ::SimpleForwardResult<FloatT, WordIdxType, EntityIdxType>(
          flattened_words->size() / window_size, /* batch_size */
          regularization_lambda),
      window_size_(window_size),
      num_random_entities_(num_random_entities),

      entity_ids_(entity_ids),

      flattened_words_(flattened_words),
      flattened_word_weights_(flattened_word_weights),

      phrase_reprs_(nullptr),
      broadcasted_instance_weights_(nullptr),
      word_projections_(nullptr),
      broadcasted_word_projections_(nullptr),
      entity_representations_(nullptr) {
    CHECK_GT(window_size_, 0);
    CHECK_EQ(flattened_words->size() % window_size, 0);

    CHECK_NE(flattened_words_.get(), (device_matrix<WordIdxType>*) nullptr);
    CHECK_DIMENSIONS(*flattened_words_, 1, this->batch_size_ * window_size_);

    CHECK_DIMENSIONS_EQUAL(*flattened_words_, *flattened_word_weights_);

    CHECK_NE(entity_ids_.get(), (device_matrix<EntityIdxType>*) nullptr);

    CHECK_DIMENSIONS(*entity_ids_, 1, entity_ids_->size());
}

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
ForwardResult<FloatT, WordIdxType, EntityIdxType>::ForwardResult()
    : ::SimpleForwardResult<FloatT, WordIdxType, EntityIdxType>(),
      window_size_(0), num_random_entities_(0) {}

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
bool ForwardResult<FloatT, WordIdxType, EntityIdxType>::complete() const {
    #ifdef NDEBUG
        LOG_EVERY_N(ERROR, 10000000000000000000000)
            << "Calling this function is non-debug mode can decrease performance.";
    #endif

    return !std::isnan(this->get_cost()) &&
        phrase_reprs_ != nullptr &&
        broadcasted_instance_weights_ != nullptr &&
        word_projections_ != nullptr &&
        broadcasted_word_projections_ != nullptr &&
        entity_representations_ != nullptr &&
        this->similarity_probs_ != nullptr &&
        this->pointwise_mass_ != nullptr;
}

// Explicit instantiations.
template class ForwardResult<FLOATING_POINT_TYPE, int32, int32>;

}  // namespace TextEntity

namespace RepresentationSimilarity {

template <typename FloatT, typename ReprIdxType>
ForwardResult<FloatT, ReprIdxType>::ForwardResult(
        const ParamIdentifier param_id,
        device_matrix<ReprIdxType>* const ids,
        device_matrix<FloatT>* const weights,
        const FloatT regularization_lambda)
    : ::SimpleForwardResult<FloatT, ReprIdxType, ReprIdxType>(
          ids->size() / 2, /* batch_size */
          regularization_lambda),
      param_id_(param_id),
      ids_(ids),
      weights_(weights) {
    CHECK_EQ(ids->size() % 2, 0);
    CHECK_EQ(weights->size() * 2, ids_->size());
}

// For testing.
template <typename FloatT, typename ReprIdxType>
ForwardResult<FloatT, ReprIdxType>::ForwardResult(const ParamIdentifier param_id)
    : ::SimpleForwardResult<FloatT, ReprIdxType, ReprIdxType>(),
      param_id_(param_id) {}

template <typename FloatT, typename ReprIdxType>
bool ForwardResult<FloatT, ReprIdxType>::complete() const {
    #ifdef NDEBUG
        LOG_EVERY_N(ERROR, 10000000000000000000000)
            << "Calling this function is non-debug mode can decrease performance.";
    #endif

    return !std::isnan(this->get_cost()) &&
        ids_ != nullptr &&
        weights_ != nullptr &&
        representations_ != nullptr &&
        this->similarity_probs_ != nullptr &&
        this->pointwise_mass_ != nullptr;
}

template <typename FloatT, typename ReprIdxType>
device_matrix<ReprIdxType>* ForwardResult<FloatT, ReprIdxType>::get_indices(const ParamIdentifier param_id) const {
    if (param_id_ == param_id) {
        return ids_.get();
    } else {
        return nullptr;
    }
}

// Explicit instantiations.
template class ForwardResult<FLOATING_POINT_TYPE, int32>;

}  // namespace RepresentationSimilarity

// Explicit instantiations.
template class Gradients<FLOATING_POINT_TYPE>;
template class SingleGradients<FLOATING_POINT_TYPE>;
template class CompositeGradients<FLOATING_POINT_TYPE>;

template class MultiForwardResultBase<
    FLOATING_POINT_TYPE, int32, int32,
    TextEntity::ForwardResult<FLOATING_POINT_TYPE, int32, int32>,
    RepresentationSimilarity::ForwardResult<FLOATING_POINT_TYPE, int32>>;
template class MergeGradientsFn<FLOATING_POINT_TYPE>;