#ifndef CUNVSM_INTERMEDIATE_RESULTS_H
#define CUNVSM_INTERMEDIATE_RESULTS_H

#include <glog/logging.h>
#include <gtest/gtest_prod.h>

#include "cuNVSM/base.h"
#include "cuNVSM/cuda_utils.h"
#include "cuNVSM/cudnn_utils.h"

// Forward declarations.
template <typename FloatT>
class Gradients;
template <typename FloatT>
class SingleGradients;
template <typename FloatT>
class CompositeGradients;
template <typename FloatT, typename WordIdxT, typename EntityIdxT>
class ForwardResult;
template <typename FloatT, typename WordIdxT, typename EntityIdxT>
class SimpleForwardResult;

template <typename FloatT>
class MergeGradientsFn;

#include "params.h"
#include "storage.h"

class ExtractGradient;

template <typename ModelT>
class GradientCheckFn;

// Forward declaration for testing.
class ParamsTest_Representations_update_Test;
class ParamsTest_Transform_BatchNormalization_Test;

namespace TextEntity {
class Objective;
}  // namespace TextEntity
namespace RepresentationSimilarity {
class Objective;
}  // namespace RepresentationSimilarity

template <typename FloatT>
class Gradients {
 public:
  typedef std::unique_ptr<device_matrix<FloatT>> GradientType;

  virtual typename TransformStorage<FloatT>::GradientType* get_transform_gradient(
      const ParamIdentifier param_id) const;

  virtual typename RepresentationsStorage<FloatT, int32>::GradientType* get_representations_gradient(
      const ParamIdentifier param_id) const = 0;

  virtual ~Gradients() {}

 protected:
  Gradients()
      : grad_entity_repr_(nullptr),
        grad_phrase_reprs_(nullptr),
        grad_transform_matrix_(nullptr),
        grad_bias_(nullptr),
        grads_({&grad_entity_repr_,
                &grad_phrase_reprs_,
                &grad_transform_matrix_,
                &grad_bias_}) {}

  GradientType grad_entity_repr_;
  GradientType grad_phrase_reprs_;
  GradientType grad_transform_matrix_;
  GradientType grad_bias_;

  const std::vector<GradientType*> grads_;

  friend class TextEntity::Objective;
  friend class RepresentationSimilarity::Objective;

  template <typename ModelT>
  friend class GradientCheckFn;

  template <typename FloatT_>
  friend class MergeGradientsFn;

  // For testing purposes.
  FRIEND_TEST(ParamsTest, Representations_update);
  FRIEND_TEST(ParamsTest, Transform_backward);
  FRIEND_TEST(ParamsTest, Transform_BatchNormalization);

 private:
  DISALLOW_COPY_AND_ASSIGN(Gradients);
};

template <typename FloatT>
class SingleGradients : public Gradients<FloatT> {
 public:
  virtual typename RepresentationsStorage<FloatT, int32>::GradientType* get_representations_gradient(
      const ParamIdentifier param_id) const override;

  explicit SingleGradients(const typename Storage<FloatT>::ForwardResult* const result)
      : Gradients<FloatT>(), result_(result) {}

  virtual ~SingleGradients() {}

 protected:
  const typename Storage<FloatT>::ForwardResult* const result_;

 private:
  DISALLOW_COPY_AND_ASSIGN(SingleGradients);
};

template <typename FloatT>
class CompositeGradients : public Gradients<FloatT> {
 public:
  virtual typename RepresentationsStorage<FloatT, int32>::GradientType* get_representations_gradient(
      const ParamIdentifier param_id) const;

  virtual ~CompositeGradients() {}

 protected:
  // Takes ownership.
  CompositeGradients(const std::vector<Gradients<FloatT>*>& constituent_gradients)
      : Gradients<FloatT>(),
        constituent_gradients_() {
      for (Gradients<FloatT>* const grad : constituent_gradients) {
          constituent_gradients_.push_back(
              std::move(std::unique_ptr<Gradients<FloatT>>(grad)));
      }
  }

  std::vector<std::unique_ptr<Gradients<FloatT>>> constituent_gradients_;

  friend class MergeGradientsFn<FloatT>;

  DISALLOW_COPY_AND_ASSIGN(CompositeGradients);
};

template <typename FloatT>
class MergeGradientsFn {
 public:
  typedef std::pair<Gradients<FloatT>*, FloatT> GradientAndWeight;
  typedef std::vector<GradientAndWeight> GradientAndWeights;

  // Takes ownership.
  Gradients<FloatT>* operator()(const GradientAndWeights& gradients_and_weights) const;
};

template <typename FloatT, typename WordIdxT, typename EntityIdxT>
class ForwardResult {
 public:
  typedef WordIdxT WordIdxType;
  typedef EntityIdxT EntityIdxType;

  ForwardResult() {}

  virtual ~ForwardResult() {}

  virtual FloatT get_cost() const = 0;
  virtual FloatT scaled_regularization_lambda() const = 0;

  virtual device_matrix<FloatT>* get_similarity_probs() const = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(ForwardResult);
};

template <typename FloatT, typename WordIdxT, typename EntityIdxT>
class SimpleForwardResult : public ForwardResult<FloatT, WordIdxT, EntityIdxT> {
 public:
  typedef WordIdxT WordIdxType;
  typedef EntityIdxT EntityIdxType;

  virtual FloatT get_cost() const;
  virtual FloatT scaled_regularization_lambda() const;

  virtual device_matrix<FloatT>* get_similarity_probs() const {
      return similarity_probs_.get();
  }

  virtual size_t get_window_size() const = 0;

  virtual device_matrix<WordIdxType>* get_word_indices() const = 0;
  virtual device_matrix<EntityIdxType>* get_entity_indices() const = 0;

  virtual device_matrix<FloatT>* get_word_weights() const = 0;

 protected:
  SimpleForwardResult(const size_t batch_size,
                      const FloatT regularization_lambda);

  SimpleForwardResult();

  std::unique_ptr<device_matrix<FloatT>> pointwise_mass_;
  std::unique_ptr<device_matrix<FloatT>> similarity_probs_;

  FloatT cost_;

  virtual bool complete() const = 0;

  const size_t batch_size_;
  const FloatT regularization_lambda_;

 private:
  DISALLOW_COPY_AND_ASSIGN(SimpleForwardResult);
};

template <typename FloatT, typename WordIdxT, typename EntityIdxT, typename ... ForwardResultT>
class MultiForwardResultBase : public ForwardResult<FloatT, WordIdxT, EntityIdxT> {
 public:
  typedef WordIdxT WordIdxType;
  typedef EntityIdxT EntityIdxType;

  typedef std::tuple<std::pair<ForwardResultT*, FloatT> ...> ForwardResultsType;

  explicit MultiForwardResultBase(const ForwardResultsType& forward_results);

  virtual FloatT get_cost() const;
  virtual FloatT scaled_regularization_lambda() const;

  virtual device_matrix<FloatT>* get_similarity_probs() const;

  // TODO(cvangysel): make protected.
  std::tuple<std::pair<std::unique_ptr<ForwardResultT>, FloatT> ...> forward_results_;

 protected:
  std::unique_ptr<device_matrix<FloatT>> similarity_probs_;

 private:
  DISALLOW_COPY_AND_ASSIGN(MultiForwardResultBase);
};

namespace TextEntity {

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
class ForwardResult : public ::SimpleForwardResult<FloatT, WordIdxType, EntityIdxType> {
 public:
  virtual ~ForwardResult() {}

  virtual device_matrix<WordIdxType>* get_word_indices() const {
      return flattened_words_.get();
  }

  virtual device_matrix<EntityIdxType>* get_entity_indices() const {
      return entity_ids_.get();
  }

  virtual device_matrix<FloatT>* get_word_weights() const {
      return flattened_word_weights_.get();
  }

  virtual size_t get_window_size() const {
      return window_size_;
  }

 protected:
  ForwardResult(
      device_matrix<WordIdxType>* const flattened_words,
      device_matrix<FloatT>* const flattened_word_weights,
      device_matrix<EntityIdxType>* const entity_ids,
      const size_t window_size,
      const size_t num_random_entities,
      const FloatT regularization_lambda);

  // For testing.
  ForwardResult();

  bool complete() const override;

  const size_t window_size_;
  const size_t num_random_entities_;

  std::unique_ptr<device_matrix<EntityIdxType>> entity_ids_;

  std::unique_ptr<device_matrix<WordIdxType>> flattened_words_;
  std::unique_ptr<device_matrix<FloatT>> flattened_word_weights_;

  std::unique_ptr<device_matrix<FloatT>> phrase_reprs_;

  std::unique_ptr<device_matrix<FloatT>> broadcasted_instance_weights_;

  std::unique_ptr<device_matrix<FloatT>> word_projections_;
  std::unique_ptr<device_matrix<FloatT>> broadcasted_word_projections_;
  std::unique_ptr<device_matrix<FloatT>> entity_representations_;

  // Per-instance l2-normalization.
  std::unique_ptr<Normalizer<FloatT>> phrase_normalizer_;
  std::unique_ptr<Normalizer<FloatT>> entity_normalizer_;

  // Per-feature standardization.
  std::unique_ptr<BatchNormalization<FloatT>> batch_normalization_;

  friend class ::Gradients<FloatT>;

  friend class Objective;

  // Remove Transform<FloatT> as friend.
  friend class Transform<FloatT>;

  template <typename ModelT>
  friend class GradientCheckFn;

  // For testing purposes.
  FRIEND_TEST(::ParamsTest, Representations_update);
  FRIEND_TEST(::ParamsTest, Transform_BatchNormalization);

  DISALLOW_COPY_AND_ASSIGN(ForwardResult);
};

}  // namespace TextEntity

namespace RepresentationSimilarity {

template <typename FloatT, typename ReprIdxType>
class ForwardResult : public ::SimpleForwardResult<FloatT, ReprIdxType, ReprIdxType> {
 public:
  virtual ~ForwardResult() {}

  virtual device_matrix<ReprIdxType>* get_indices(const ParamIdentifier param_id) const;

  virtual device_matrix<ReprIdxType>* get_word_indices() const {
      return get_indices(WORD_REPRS);
  }

  virtual device_matrix<ReprIdxType>* get_entity_indices() const {
      return get_indices(ENTITY_REPRS);
  }

  virtual device_matrix<FloatT>* get_word_weights() const {
      return nullptr;
  }

  virtual size_t get_window_size() const {
      return 1;
  }

 protected:
  ForwardResult(const ParamIdentifier param_id,
                device_matrix<ReprIdxType>* const ids,
                device_matrix<FloatT>* const weights,
                const FloatT regularization_lambda);

  // For testing.
  ForwardResult(const ParamIdentifier param_id);

  bool complete() const override;

  const ParamIdentifier param_id_;

  std::unique_ptr<device_matrix<ReprIdxType>> ids_;
  std::unique_ptr<device_matrix<FloatT>> weights_;

  std::unique_ptr<device_matrix<FloatT>> representations_;

  friend class ::Gradients<FloatT>;

  friend class Objective;

  template <typename ModelT>
  friend class GradientCheckFn;

  DISALLOW_COPY_AND_ASSIGN(ForwardResult);
};

}  // namespace RepresentationSimilarity

#endif /* CUNVSM_INTERMEDIATE_RESULTS_H */