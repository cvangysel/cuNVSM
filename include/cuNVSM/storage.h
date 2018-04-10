#ifndef CUNVSM_STORAGE_H
#define CUNVSM_STORAGE_H

#include <map>
#include <string>

#include <glog/logging.h>
#include <gtest/gtest_prod.h>

#include "cuNVSM/base.h"
#include "cuNVSM/cuda_utils.h"

// Forward declarations.
template <typename FloatT>
class Storage;
template <typename FloatT>
class TransformStorage;
template <typename FloatT, typename IdxType>
class RepresentationsStorage;

#include "intermediate_results.h"
#include "storage_inl.h"

template <typename FloatT>
class Storage {
 public:
  typedef FloatT FloatType;

  typedef int32 WordIdxType;
  typedef int32 EntityIdxType;

  typedef SimpleForwardResult<FloatT, WordIdxType, EntityIdxType> ForwardResult;
  typedef Gradients<FloatT> Gradients;

  typedef std::map<std::string, const device_matrix<FloatT>*> DataType;

  Storage() {}
  virtual ~Storage() {}

  void initialize_with_null() {
      initialize_with_constant(0.0);
  }

  virtual DataType get_data() const = 0;

  virtual size_t num_parameters() const = 0;
  virtual void increment_parameter(const size_t idx,
                                   const FloatT epsilon) = 0;

 protected:
  virtual void initialize_with_constant(const FloatT value) = 0;

 private:
  // For testing.
  friend class UpdatesTest;

  DISALLOW_COPY_AND_ASSIGN(Storage);
};

template <typename FloatT, typename IdxType>
class RepresentationsStorage : public Storage<FloatT> {
 public:
  typedef std::tuple<device_matrix<FloatT>&, /* grad_repr */
                     const device_matrix<IdxType>&, /* repr_idx */
                     const size_t, /* window_size */
                     const device_matrix<FloatT>* /* idx_weights */> SingleGradientType;

  typedef std::vector<SingleGradientType> GradientType;

  RepresentationsStorage(const size_t num_objects,
                         const size_t size,
                         Streams* const streams);

  inline size_t repr_size() const {
      return reprs_.getRows();
  }

  inline size_t num_objects() const {
      return reprs_.getCols();
  }

  template <typename UpdateTransformOp = func::identity<FloatT>,
            typename AggOp = thrust::plus<FloatT>>
  void update(const GradientType& gradient_descs,
              const FloatT learning_rate,
              const FloatT scaled_regularization_lambda,
              Streams* const streams,
              const UpdateTransformOp update_transform_op = UpdateTransformOp(),
              const AggOp agg_op = AggOp());

  template <typename GradientIterator,
            typename UpdateTransformOp = thrust::identity<FloatT>,
            typename AggOp = thrust::plus<FloatT>>
  void update_dense(const cudaStream_t stream,
                    GradientIterator grad_reprs_it,
                    const FloatT learning_rate,
                    const FloatT scaled_regularization_lambda,
                    const UpdateTransformOp update_transform_op = UpdateTransformOp(),
                    const AggOp agg_op = AggOp()) {
      CHECK_GE(learning_rate, 0.0);
      CHECK_GE(scaled_regularization_lambda, 0.0);

      ::update_dense(
          merge_streams(stream, reprs_.getStream()),
          &reprs_,
          grad_reprs_it,
          learning_rate,
          scaled_regularization_lambda,
          update_transform_op,
          agg_op);

      CHECK_MATRIX(reprs_);
  }

  device_matrix<FloatT>* get();

  virtual typename Storage<FloatT>::DataType get_data() const override;

  virtual inline size_t num_parameters() const override {
      return reprs_.size();
  }

  virtual void increment_parameter(const size_t idx,
                                   const FloatT epsilon) override;

  FloatT get_parameter_gradient(const GradientType& gradient_descs,
                                const size_t idx) const;

 protected:
  virtual void initialize_with_constant(const FloatT value) override;

  device_matrix<FloatT> reprs_;

 private:
  // For testing purposes.
  friend class ParamsTest;
  FRIEND_TEST(ParamsTest, RepresentationsStorage_update_dense);
  FRIEND_TEST(UpdatesTest, AdamRepresentationsGradientUpdater_DENSE);
  FRIEND_TEST(UpdatesTest, AdamRepresentationsGradientUpdater_DENSE_UPDATE_DENSE_VARIANCE);

  DISALLOW_COPY_AND_ASSIGN(RepresentationsStorage);
};

template <typename FloatT>
class TransformStorage : public Storage<FloatT> {
 public:
  typedef std::tuple<device_matrix<FloatT>&, /* grad_transform */
                     device_matrix<FloatT>&, /* grad_bias */> GradientType;

  typedef std::tuple<device_matrix<FloatT>*, /* transform */
                     device_matrix<FloatT>*, /* bias */> ParamType;

  TransformStorage(const size_t word_repr_size,
                   const size_t entity_repr_size,
                   Streams* const streams);

  template <typename UpdateTransformOp = func::identity<FloatT>,
            typename AggOp = thrust::plus<FloatT>>
  void update(const GradientType& gradient_desc,
              const FloatT learning_rate,
              const FloatT scaled_regularization_lambda,
              Streams* const streams,
              const UpdateTransformOp update_transform_op = UpdateTransformOp(),
              const AggOp agg_op = AggOp());

  ParamType get();

  virtual typename Storage<FloatT>::DataType get_data() const override;

  inline virtual size_t num_parameters() const override {
      return transform_.size() + bias_.size();
  }

  virtual void increment_parameter(const size_t idx,
                                   const FloatT epsilon) override;

  FloatT get_parameter_gradient(const GradientType& gradient_desc,
                                const size_t idx) const;

 protected:
  virtual void initialize_with_constant(const FloatT value) override;

  device_matrix<FloatT> transform_;
  device_matrix<FloatT> bias_;

 private:
  DISALLOW_COPY_AND_ASSIGN(TransformStorage);
};

#endif /* CUNVSM_STORAGE_H */