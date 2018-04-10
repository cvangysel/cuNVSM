#ifndef CUNVSM_PARAMS_H
#define CUNVSM_PARAMS_H

#include <glog/logging.h>
#include <gtest/gtest_prod.h>

#include "cuNVSM/base.h"
#include "cuNVSM/cuda_utils.h"

// Forward declarations.
template <typename FloatT>
class Transform;
template <typename FloatT, typename IdxType>
class Representations;

#include "cuNVSM/intermediate_results.h"
#include "cuNVSM/storage.h"
#include "cuNVSM/updates.h"

#include "nvsm.pb.h"

// Technical debt; remove eventually.
typedef lse::TrainConfig::UpdateMethod UpdateMethod;
typedef lse::TrainConfig::UpdateMethodConf UpdateMethodConf;
#define SGD lse::TrainConfig::SGD
#define ADAGRAD lse::TrainConfig::ADAGRAD
#define ADAM lse::TrainConfig::ADAM

extern char const* ParamName[];

template <typename FloatT>
class Parameters {
 public:
  explicit Parameters(const ParamIdentifier id)
      : id_(id), initialized_(false) {}

  virtual ~Parameters() {}

  virtual void initialize(RNG* const rng) {
      initialized_ = true;
  };

  inline bool initialized() const {
      return initialized_;
  }

  virtual void update(const typename Storage<FloatT>::Gradients& gradients,
                      const FloatT learning_rate,
                      const FloatT scaled_regularization_lambda,
                      Streams* const streams) = 0;

  virtual FloatT get_parameter_gradient(
      const typename Storage<FloatT>::Gradients& gradients,
      const size_t idx) const = 0;

 protected:
  const ParamIdentifier id_;

 private:
  bool initialized_;

  DISALLOW_COPY_AND_ASSIGN(Parameters);
};

template <typename FloatT, typename IdxType>
class Representations : public Parameters<FloatT>,
                        public RepresentationsStorage<FloatT, IdxType> {
 public:
  using RepresentationsStorage<FloatT, IdxType>::reprs_;

  Representations(const ParamIdentifier id,
                  const size_t num_objects,
                  const size_t size,
                  const UpdateMethodConf& update_method,
                  Streams* const streams);

  virtual ~Representations();

  virtual void initialize(RNG* const rng) override;

  inline size_t num_objects() const {
      return reprs_.getCols();
  }

  inline size_t size() const {
      return reprs_.getRows();
  }

  device_matrix<FloatT>* get_representations(
      const cudaStream_t stream,
      const device_matrix<IdxType>& indices) const;

  // Note: slow implementation; only use for debugging.
  device_matrix<FloatT>* get_representation(const IdxType idx) const;

  device_matrix<FloatT>* get_average_representations(
      const cudaStream_t stream,
      const device_matrix<IdxType>& indices,
      const size_t window_size,
      const device_matrix<FloatT>* const indices_weights = nullptr) const;

  // Helpers for representation similarity computation.
  //
  // These implementations are not maximally optimized, due to the
  // fact that they are only used for testing and evaluation.

  device_matrix<FloatT>* compute_similarity(
      const device_matrix<FloatT>& first,
      const device_matrix<FloatT>& second) const;

  std::vector<std::vector<FloatT>> compute_similarity(
      const device_matrix<FloatT>& input_vectors,
      const std::vector<IdxType>& indices) const;

  FloatT compute_similarity(const IdxType first, const IdxType second) const;

  virtual void update(const typename Storage<FloatT>::Gradients& gradients,
                      const FloatT learning_rate,
                      const FloatT scaled_regularization_lambda,
                      Streams* const streams) override;

  virtual FloatT get_parameter_gradient(
      const typename Storage<FloatT>::Gradients& gradients,
      const size_t idx) const override;

 protected:
  std::unique_ptr<RepresentationsGradientUpdater<FloatT, IdxType>> updater_;

 private:
  // For testing purposes.
  friend class ParamsTest;
  FRIEND_TEST(ParamsTest, Representations_update);

  DISALLOW_COPY_AND_ASSIGN(Representations);
};

template <typename FloatT>
class Transform : public Parameters<FloatT>,
                  public TransformStorage<FloatT> {
 public:
  using TransformStorage<FloatT>::transform_;
  using TransformStorage<FloatT>::bias_;

  Transform(const ParamIdentifier id,
            const lse::ModelDesc::TransformDesc& desc,
            const size_t word_repr_size,
            const size_t entity_repr_size,
            const UpdateMethodConf& update_method,
            Streams* const streams);

  inline size_t source_repr_size() const {
    return transform_.getCols();
  }

  inline size_t target_repr_size() const {
    return transform_.getRows();
  }

  virtual void initialize(RNG* const rng) override;

  virtual ~Transform();

  device_matrix<FloatT>* transform(
      const cudaStream_t stream,
      const device_matrix<FloatT>& word_repr,
      BatchNormalization<FloatT>* const batch_normalization) const;

  void backward(const cudaStream_t stream,
                const typename Storage<FloatT>::ForwardResult& result,
                const device_matrix<FloatT>& broadcasted_input,
                const device_matrix<FloatT>& output,
                device_matrix<FloatT>* const grad_output,
                Gradients<FloatT>* const gradients) const;

  virtual void update(const typename Storage<FloatT>::Gradients& gradients,
                      const FloatT learning_rate,
                      const FloatT scaled_regularization_lambda,
                      Streams* const streams) override;

  virtual FloatT get_parameter_gradient(
      const typename Storage<FloatT>::Gradients& gradients,
      const size_t idx) const override;

 protected:
  const lse::ModelDesc::TransformDesc desc_;

  std::unique_ptr<TransformGradientUpdater<FloatT>> updater_;

 private:
  // For testing purposes.
  friend class ParamsTest;

  DISALLOW_COPY_AND_ASSIGN(Transform);
};

#endif /* CUNVSM_PARAMS_H */