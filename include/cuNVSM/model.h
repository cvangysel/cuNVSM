#ifndef CUNVSM_MODEL_H
#define CUNVSM_MODEL_H

#include <memory>
#include <set>
#include <type_traits>

#include <glog/logging.h>
#include <gtest/gtest_prod.h>

#include "cuNVSM/base.h"
#include "cuNVSM/cuda_utils.h"
#include "cuNVSM/data.h"
#include "cuNVSM/intermediate_results.h"
#include "cuNVSM/objective.h"
#include "cuNVSM/params.h"

#include "nvsm.pb.h"

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
class ModelBase {
 public:
  typedef Representations<FloatT, WordIdxType> WordRepresentationsType;
  typedef Representations<FloatT, EntityIdxType> EntityRepresentationsType;

  ModelBase(const size_t num_words,
            const size_t num_entities,
            const lse::ModelDesc& desc,
            const lse::TrainConfig& train_config);

  virtual ~ModelBase();

  void initialize(RNG* const rng);

  inline bool initialized() const {
      return words_.initialized() &&
          entities_.initialized() &&
          transform_.initialized();
  }

  size_t num_parameters() const {
      return words_.num_parameters() +
          entities_.num_parameters() +
          transform_.num_parameters();
  }

  device_matrix<FloatT>* get_phrase_representations(
      const cudaStream_t stream,
      const device_matrix<WordIdxType>& words,
      const size_t window_size,
      const device_matrix<FloatT>* const flattened_word_weights = nullptr) const;

  typename Storage<FloatT>::DataType get_data() const;

 protected:
  const lse::ModelDesc desc_;

  std::unique_ptr<Streams> streams_;

  Representations<FloatT, WordIdxType> words_;
  Representations<FloatT, EntityIdxType> entities_;

  Transform<FloatT> transform_;

  const std::map<ParamIdentifier, Parameters<FloatT>*> params_;

  friend class TextEntity::Objective;
  friend class RepresentationSimilarity::Objective;

  // For gradient checking.
  template <typename ModelT>
  friend class GradientCheckFn;
};

template <typename ObjectiveT>
class Model : public ModelBase<typename ObjectiveT::FloatT,
                               typename ObjectiveT::WordIdxType,
                               typename ObjectiveT::EntityIdxType> {
 public:
  typedef ObjectiveT Objective;

  Model(const size_t num_words,
        const size_t num_entities,
        const lse::ModelDesc& desc,
        const lse::TrainConfig& train_config);

  typedef typename Objective::WordIdxType WordIdxType;
  typedef typename Objective::EntityIdxType EntityIdxType;

  typedef typename Objective::FloatT FloatT;

  typedef typename Objective::BatchType Batch;

  typedef typename Objective::ForwardResultType ForwardResult;
  typedef typename Objective::GradientsType Gradients;

  device_matrix<FloatT>* infer(
      const std::vector<std::vector<WordIdxType>>& words,
      const size_t window_size) const;

  ForwardResult* compute_cost(const Batch& batch,
                              RNG* const rng) const;

  FloatT get_cost(const Batch& batch,
                  const std::stringstream* const rng_state,
                  RNG* const rng) const;

  void backprop(const ForwardResult& result,
                const FloatT learning_rate);

  Gradients* compute_gradients(const ForwardResult& result);

  void update(const Gradients& gradients,
              const FloatT learning_rate,
              const FloatT scaled_regularization_lambda);

 protected:
  std::unique_ptr<Objective> objective_;

  friend class TextEntity::Objective;
  friend class RepresentationSimilarity::Objective;

  // For gradient checking.
  template <typename ModelT>
  friend class GradientCheckFn;

  // For testing purposes.
  FRIEND_TEST(ParamsTest, generate_labels);

  DISALLOW_COPY_AND_ASSIGN(Model);
};

typedef Model<TextEntity::Objective> LSE;

#include "objective.h"

#endif /* CUNVSM_MODEL_H */