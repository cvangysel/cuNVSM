#ifndef CUNVSM_UPDATES_H
#define CUNVSM_UPDATES_H

#include <boost/fusion/adapted/boost_tuple.hpp>
#include <boost/fusion/include/boost_tuple.hpp>

#include <utility>

#include "cuNVSM/base.h"
#include "cuNVSM/cuda_utils.h"

// Forward declarations.
template <typename FloatT>
class TransformGradientUpdater;
template <typename FloatT, typename IdxType>
class RepresentationsGradientUpdater;

#include "cuNVSM/storage.h"
#include "nvsm.pb.h"

#define DEFAULT_EPSILON 1e-6

template <typename FloatT>
void apply_regularization(
        const cudaStream_t stream,
        const FloatT scaled_regularization_lambda,
        const device_matrix<FloatT>* const param,
        device_matrix<FloatT>* const grad_param) {
    CHECK_DIMENSIONS_EQUAL(*param, *grad_param);

    // grad_param = grad_param' - scaled_regularization_lambda * param

    elemwise_plus(thrust::cuda::par.on(stream),
                  *param,
                  grad_param,
                  func::scale_by_constant<FloatT>(-scaled_regularization_lambda) /* first_op_op */);
}

template <typename FloatT>
void apply_regularization(
        const cudaStream_t stream,
        const FloatT scaled_regularization_lambda,
        TransformStorage<FloatT>* const storage,  // TODO(cvangysel): should be const.
        typename TransformStorage<FloatT>::GradientType* const gradient_desc) {
    typename TransformStorage<FloatT>::ParamType storage_data = storage->get();

    // TODO(cvangysel): allow this to generalize with compile-time voodoo.
    // constexpr size_t num_params =
    //     std::tuple_size<typename TransformStorage<FloatT>::ParamType>::value;

    apply_regularization(stream,
                         scaled_regularization_lambda,
                         std::get<0>(storage_data),
                         &std::get<0>(*gradient_desc));

    // Do not apply regularization on bias.
    //
    // apply_regularization(stream,
    //                      scaled_regularization_lambda,
    //                      std::get<1>(storage_data),
    //                      &std::get<1>(*gradient_desc));
}

template <typename FloatT>
class GradientUpdater {
 public:
  // Takes ownership of pointers in storages.
  GradientUpdater(const FloatT epsilon, const std::vector<Storage<FloatT>*>& storages);
  virtual ~GradientUpdater() {}

 protected:
  const FloatT epsilon_;
  std::vector<std::unique_ptr<Storage<FloatT>>> storages_;

 private:
  // For testing purposes.
  FRIEND_TEST(UpdatesTest, AdagradTransformGradientUpdater);
  FRIEND_TEST(UpdatesTest, AdagradRepresentationsGradientUpdater);
  FRIEND_TEST(UpdatesTest, AdamTransformGradientUpdater);

  FRIEND_TEST(UpdatesTest, AdamRepresentationsGradientUpdater_SPARSE);
  FRIEND_TEST(UpdatesTest, AdamRepresentationsGradientUpdater_DENSE_UPDATE);
  FRIEND_TEST(UpdatesTest, AdamRepresentationsGradientUpdater_DENSE_UPDATE_DENSE_VARIANCE);
};

template <typename FloatT>
class TransformGradientUpdater : public GradientUpdater<FloatT> {
 public:
  // Takes ownership of pointers in storages.
  TransformGradientUpdater(const FloatT epsilon,
                           const std::vector<Storage<FloatT>*>& storages)
      : GradientUpdater<FloatT>(epsilon, storages) {}

  virtual void update(TransformStorage<FloatT>* const storage,
                      typename TransformStorage<FloatT>::GradientType* const gradient_desc,
                      const FloatT learning_rate,
                      const FloatT scaled_regularization_lambda,
                      Streams* const streams) = 0;
};

template <typename FloatT, typename IdxType>
class RepresentationsGradientUpdater : public GradientUpdater<FloatT> {
 public:
  // Takes ownership of pointers in storages.
  RepresentationsGradientUpdater(const FloatT epsilon,
                                 const std::vector<Storage<FloatT>*>& storages)
      : GradientUpdater<FloatT>(epsilon, storages) {}

  virtual void update(RepresentationsStorage<FloatT, IdxType>* const storage,
                      typename RepresentationsStorage<FloatT, IdxType>::GradientType* const gradient_desc,
                      const FloatT learning_rate,
                      const FloatT scaled_regularization_lambda,
                      Streams* const streams) = 0;
};

// SGD.

template <typename FloatT>
class SGDTransformGradientUpdater : public TransformGradientUpdater<FloatT> {
 public:
  SGDTransformGradientUpdater()
      : TransformGradientUpdater<FloatT>(DEFAULT_EPSILON, {}) {}

  virtual void update(TransformStorage<FloatT>* const storage,
                      typename TransformStorage<FloatT>::GradientType* const gradient_desc,
                      const FloatT learning_rate,
                      const FloatT scaled_regularization_lambda,
                      Streams* const streams) override;
};

template <typename FloatT, typename IdxType>
class SGDRepresentationsGradientUpdater : public RepresentationsGradientUpdater<FloatT, IdxType> {
 public:
  SGDRepresentationsGradientUpdater()
      : RepresentationsGradientUpdater<FloatT, IdxType>(DEFAULT_EPSILON, {}) {}

  virtual void update(RepresentationsStorage<FloatT, IdxType>* const storage,
                      typename RepresentationsStorage<FloatT, IdxType>::GradientType* const gradient_desc,
                      const FloatT learning_rate,
                      const FloatT scaled_regularization_lambda,
                      Streams* const streams) override;
};

// Adagrad.

template <typename FloatT>
class AdagradTransformGradientUpdater : public TransformGradientUpdater<FloatT> {
 public:
  AdagradTransformGradientUpdater(const size_t source_vector_dim,
                                  const size_t target_vector_dim,
                                  Streams* const streams,
                                  const FloatT epsilon = DEFAULT_EPSILON);

  virtual void update(TransformStorage<FloatT>* const storage,
                      typename TransformStorage<FloatT>::GradientType* const gradient_desc,
                      const FloatT learning_rate,
                      const FloatT scaled_regularization_lambda,
                      Streams* const streams) override;
};

template <typename FloatT, typename IdxType>
class AdagradRepresentationsGradientUpdater : public RepresentationsGradientUpdater<FloatT, IdxType> {
 public:
  AdagradRepresentationsGradientUpdater(const size_t num_objects,
                                        Streams* const streams,
                                        const FloatT epsilon = DEFAULT_EPSILON);

  virtual void update(RepresentationsStorage<FloatT, IdxType>* const storage,
                      typename RepresentationsStorage<FloatT, IdxType>::GradientType* const gradient_desc,
                      const FloatT learning_rate,
                      const FloatT scaled_regularization_lambda,
                      Streams* const streams) override;
};

// Adam.

template <typename FloatT>
class AdamTransformGradientUpdater : public TransformGradientUpdater<FloatT> {
 public:
  AdamTransformGradientUpdater(const size_t source_vector_dim,
                               const size_t target_vector_dim,
                               Streams* const streams,
                               const FloatT beta1 = 0.9,
                               const FloatT beta2 = 0.999,
                               const FloatT epsilon = DEFAULT_EPSILON);

  virtual void update(TransformStorage<FloatT>* const storage,
                      typename TransformStorage<FloatT>::GradientType* const gradient_desc,
                      const FloatT learning_rate,
                      const FloatT scaled_regularization_lambda,
                      Streams* const streams) override;

 protected:
  const FloatT beta1_;
  const FloatT beta2_;

  uint64 t_;
};

typedef lse::TrainConfig_UpdateMethodConf_AdamConf AdamConf;

template <typename FloatT, typename IdxType>
class AdamRepresentationsGradientUpdater : public RepresentationsGradientUpdater<FloatT, IdxType> {
 public:
  AdamRepresentationsGradientUpdater(const size_t num_objects,
                                     const size_t repr_size,
                                     const AdamConf& conf,
                                     Streams* const streams,
                                     const FloatT beta1 = 0.9,
                                     const FloatT beta2 = 0.999,
                                     const FloatT epsilon = DEFAULT_EPSILON);

  virtual void update(RepresentationsStorage<FloatT, IdxType>* const storage,
                      typename RepresentationsStorage<FloatT, IdxType>::GradientType* const gradient_desc,
                      const FloatT learning_rate,
                      const FloatT scaled_regularization_lambda,
                      Streams* const streams) override;

 protected:
  const AdamConf conf_;

  const FloatT beta1_;
  const FloatT beta2_;

  uint64 t_;
};

#endif /* CUNVSM_UPDATES_H */