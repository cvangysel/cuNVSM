#ifndef CUNVSM_OBJECTIVE_H
#define CUNVSM_OBJECTIVE_H

#include <memory>
#include <set>
#include <type_traits>

#include <glog/logging.h>
#include <gtest/gtest_prod.h>

#include "cuNVSM/base.h"
#include "cuNVSM/model.h"
#include "cuNVSM/cuda_utils.h"
#include "cuNVSM/intermediate_results.h"
#include "cuNVSM/labels.h"

// Forward declarations.
template <typename FloatT, typename WordIdxType, typename EntityIdxType>
class ModelBase;

// Forward declaration for testing.
class ParamsTest_generate_labels_Test;

class Typedefs {
 public:
  typedef int32 IdxType;

  typedef IdxType WordIdxType;
  typedef IdxType EntityIdxType;

  typedef FLOATING_POINT_TYPE FloatT;

  typedef ModelBase<FloatT, WordIdxType, EntityIdxType> ModelBase;
};

template <typename BatchT, typename ForwardResultT, typename GradientsT>
class Objective {
 public:
  typedef Typedefs::WordIdxType WordIdxType;
  typedef Typedefs::EntityIdxType EntityIdxType;

  typedef Typedefs::FloatT FloatT;

  typedef BatchT BatchType;

  typedef ForwardResultT ForwardResultType;
  typedef GradientsT GradientsType;

  Objective(::Typedefs::ModelBase* const model,
            const lse::TrainConfig& train_config)
          : model_(model), train_config_(train_config) {
      CHECK_NOTNULL(model);
  }

  virtual ~Objective() {}

  virtual ForwardResultType* compute_cost(
      const BatchType& batch, RNG* const rng) const = 0;

  virtual GradientsType* compute_gradients(
      const ForwardResultType& result) = 0;

 protected:
  const lse::TrainConfig train_config_;
  ::Typedefs::ModelBase* const model_;
};

namespace TextEntity {

class Objective : public ::Objective<Batch,
                                     ForwardResult<::Typedefs::FloatT,
                                                   ::Typedefs::WordIdxType,
                                                   ::Typedefs::EntityIdxType>,
                                     Gradients<::Typedefs::FloatT>> {
 public:
  Objective(::Typedefs::ModelBase* const model,
            const lse::TrainConfig& train_config)
      : ::Objective<Batch, ForwardResultType, GradientsType>(model, train_config),
        label_generator_(new UniformLabelGenerator<FloatT, EntityIdxType>) {}

  virtual ForwardResultType* compute_cost(const Batch& batch, RNG* const rng) const;

  virtual GradientsType* compute_gradients(const ForwardResultType& result);

 protected:
  void generate_labels(
      const EntityIdxType* const labels,
      const size_t num_labels,
      const size_t num_negative_labels,
      std::vector<EntityIdxType>* const instance_entities,
      RNG* const rng) const;

  std::unique_ptr<LabelGenerator<FloatT, EntityIdxType>> label_generator_;

  // For testing purposes.
  FRIEND_TEST(::ParamsTest, generate_labels);

  DISALLOW_COPY_AND_ASSIGN(Objective);
};

}  // namespace TextEntity

namespace RepresentationSimilarity {

class Objective : public ::Objective<Batch,
                                     ForwardResult<::Typedefs::FloatT,
                                                   ::Typedefs::IdxType>,
                                     ::Gradients<::Typedefs::FloatT>> {
 public:
  Objective(const ParamIdentifier param_id,
            ::Typedefs::ModelBase* const model,
            const lse::TrainConfig& train_config)
      : ::Objective<Batch, ForwardResultType, GradientsType>(model, train_config),
        param_id_(param_id) {
      CHECK(param_id_ == WORD_REPRS || param_id_ == ENTITY_REPRS);
  }

  virtual ForwardResultType* compute_cost(const Batch& batch, RNG* const rng) const;

  virtual GradientsType* compute_gradients(const ForwardResultType& result);

 protected:
  Representations<::Typedefs::FloatT, ::Typedefs::IdxType>* get_representation_storage() const;

  void reset_grad(GradientsType* const gradients, device_matrix<FloatT>* const grad_reprs) const;

  const ParamIdentifier param_id_;

  // For testing purposes.
  FRIEND_TEST(::ParamsTest, generate_labels);

  DISALLOW_COPY_AND_ASSIGN(Objective);
};

}  // namespace RepresentationSimilarity

namespace EntityEntity {

using RepresentationSimilarity::DataSource;
using RepresentationSimilarity::InstanceT;

class Objective : public RepresentationSimilarity::Objective {
 public:
  Objective(::Typedefs::ModelBase* const model,
            const lse::TrainConfig& train_config)
      : RepresentationSimilarity::Objective(ENTITY_REPRS, model, train_config) {}
};

}  // namespace EntityEntity

namespace TermTerm {

using RepresentationSimilarity::Batch;
using RepresentationSimilarity::DataSource;
using RepresentationSimilarity::InstanceT;

class Objective : public RepresentationSimilarity::Objective {
 public:
  Objective(::Typedefs::ModelBase* const model,
            const lse::TrainConfig& train_config)
      : RepresentationSimilarity::Objective(WORD_REPRS, model, train_config) {}
};

}  // namespace TermTerm

namespace TextEntityEntityEntity {

class Objective : public ::Objective<std::tuple<TextEntity::Batch,
                                                EntityEntity::Batch>,
                                     MultiForwardResultBase<::Typedefs::FloatT,
                                                            ::Typedefs::IdxType,
                                                            ::Typedefs::EntityIdxType,
                                                            TextEntity::ForwardResult<
                                                                ::Typedefs::FloatT,
                                                                ::Typedefs::IdxType,
                                                                ::Typedefs::EntityIdxType>,
                                                            RepresentationSimilarity::ForwardResult<
                                                                ::Typedefs::FloatT,
                                                                ::Typedefs::EntityIdxType>>,
                                     ::Gradients<::Typedefs::FloatT>> {
 public:
  Objective(::Typedefs::ModelBase* const model,
                 const lse::TrainConfig& train_config);

  virtual ForwardResultType* compute_cost(const BatchType& batch, RNG* const rng) const;

  virtual GradientsType* compute_gradients(const ForwardResultType& result);

 protected:
  // For testing purposes.
  FRIEND_TEST(::ParamsTest, generate_labels);

  const FloatT text_entity_weight_;
  const FloatT entity_entity_weight_;

  TextEntity::Objective text_entity_objective_;
  RepresentationSimilarity::Objective entity_entity_objective_;

  DISALLOW_COPY_AND_ASSIGN(Objective);
};

}  // namespace TextEntityEntityEntity

namespace TextEntityTermTerm {

class Objective : public ::Objective<std::tuple<TextEntity::Batch,
                                                EntityEntity::Batch>,
                                     MultiForwardResultBase<::Typedefs::FloatT,
                                                            ::Typedefs::IdxType,
                                                            ::Typedefs::IdxType,
                                                            TextEntity::ForwardResult<
                                                                ::Typedefs::FloatT,
                                                                ::Typedefs::IdxType,
                                                                ::Typedefs::IdxType>,
                                                            RepresentationSimilarity::ForwardResult<
                                                                ::Typedefs::FloatT,
                                                                ::Typedefs::IdxType>>,
                                     ::Gradients<::Typedefs::FloatT>> {
 public:
  Objective(::Typedefs::ModelBase* const model,
                 const lse::TrainConfig& train_config);

  virtual ForwardResultType* compute_cost(const BatchType& batch, RNG* const rng) const;

  virtual GradientsType* compute_gradients(const ForwardResultType& result);

 protected:
  // For testing purposes.
  FRIEND_TEST(::ParamsTest, generate_labels);

  const FloatT text_entity_weight_;
  const FloatT term_term_weight_;

  TextEntity::Objective text_entity_objective_;
  RepresentationSimilarity::Objective term_term_objective_;

  DISALLOW_COPY_AND_ASSIGN(Objective);
};

}  // namespace TextEntityEntityEntity

#endif /* CUNVSM_OBJECTIVE_H */