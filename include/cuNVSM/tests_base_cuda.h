#ifndef CUNVSM_TESTS_BASE_CUDA_H
#define CUNVSM_TESTS_BASE_CUDA_H

#include "cuNVSM/gradient_check.h"
#include "cuNVSM/model.h"
#include "cuNVSM/tests_base.h"

template <typename ModelT>
class ModelTestWrapper : public ModelT {
 public:
  typedef ModelT Wrapped;

  // Inherit constructor.
  using ModelT::ModelT;

  const typename ModelT::WordRepresentationsType& getWords() const {
      return this->words_;
  }

  const typename ModelT::EntityRepresentationsType& getEntities() const {
      return this->entities_;
  }
};

template <typename ModelT>
class ModelTest : public ::testing::TestWithParam<::testing::tuple<RNG::result_type,
                                                                   std::string,
                                                                   std::string>> {
 protected:
  typedef DataSource<typename ModelT::Batch> DataSourceType;

  ModelTest() : model_(nullptr), data_(nullptr) {}

  virtual void SetUp() override {
      LOG(ERROR) << SHOW_DEFINE(FLOATING_POINT_TYPE);
      rng_.seed(std::get<0>(GetParam()) + 1);
  }

  virtual typename DataSourceType::BatchType* create_batch(
      const size_t batch_size,
      const size_t window_size) const = 0;

  template <typename T>
  void clear_batch(T* const batch) const {
      batch->clear();
  }

  template <typename T>
  bool batch_full(T* const batch) const {
      return batch->full();
  }

  template <typename T>
  size_t batch_num_instances(T* const batch) const {
    return batch->num_instances();
  }

  template <typename ... T>
  void clear_batch(std::tuple<T ...>* const batch) const {
      std::get<0>(*batch).clear();
      std::get<1>(*batch).clear();
  }

  template <typename ... T>
  bool batch_full(std::tuple<T ...>* const batch) const {
      return std::get<0>(*batch).full() && std::get<1>(*batch).full();
  }

  template <typename ... T>
  size_t batch_num_instances(std::tuple<T ...>* const batch) const {
    return min(std::get<0>(*batch).num_instances(),
               std::get<1>(*batch).num_instances());
  }

  void setup_data(DataSourceType* const source) {
      data_.reset(source);
  }

  lse::ModelDesc get_desc() const {
      return ParseProto<lse::ModelDesc>(std::get<1>(GetParam()));
  }

  void setup_model(const size_t num_words, const size_t word_repr_size,
                   const size_t num_objects, const size_t object_repr_size,
                   const size_t num_random_entities, const FloatT regularization_lambda) {
      lse::ModelDesc model_desc = get_desc();

      model_desc.set_word_repr_size(word_repr_size);
      model_desc.set_entity_repr_size(object_repr_size);

      lse::TrainConfig train_config;

      train_config.set_num_random_entities(num_random_entities);
      train_config.set_regularization_lambda(regularization_lambda);

      train_config.mutable_update_method()->MergeFrom(
          ParseProto<UpdateMethodConf>(std::get<2>(GetParam())));

      // Objective weights.
      train_config.set_text_entity_weight(1.0);
      train_config.set_entity_entity_weight(1.0);
      train_config.set_term_term_weight(1.0);

      model_.reset(new ModelT(num_words, num_objects,
                              model_desc, train_config));
      model_->initialize(&rng_);
  }

  std::vector<typename ModelT::FloatT> compute_cost(
          const size_t batch_size,
          const size_t window_size) {
      return train(batch_size,
                   window_size,
                   false, /* check_gradients */
                   false /* backpropagate */);
  }

  FloatT get_learning_rate() const {
      switch (ParseProto<UpdateMethodConf>(std::get<2>(GetParam())).type()) {
          case SGD:
              return 0.1;
          case ADAGRAD:
              return 0.01;
          case ADAM:
              return 0.001;
      }

      LOG(FATAL) << "Invalid update method.";
      throw 0;
  }

  std::vector<typename ModelT::FloatT> train(
          const size_t batch_size,
          const size_t window_size,
          const bool check_gradients = false,
          const bool backpropagate = true) {
      CHECK_NE(data_.get(), (DataSourceType*) nullptr);
      CHECK_NE(model_.get(), (ModelT*) nullptr);

      std::vector<typename ModelT::FloatT> costs;

      std::unique_ptr<typename ModelT::Batch> batch(
          this->create_batch(batch_size, window_size));

      while (data_->has_next()) {
          clear_batch(batch.get());

          data_->next(batch.get());

          if (!batch_full(batch.get())) {
              continue;
          }

          if (batch_num_instances(batch.get()) % MAX_THREADS_PER_BLOCK != 0) {
              LOG_EVERY_N(WARNING, 1000)
                  << "Batch is not a multiple of "
                  << MAX_THREADS_PER_BLOCK << ".";
          }

          // Save RNG state at beginning of epoch.
          std::stringstream rng_state;
          rng_state << rng_;

          std::unique_ptr<typename ModelT::ForwardResult> result(
              model_->compute_cost(*batch, &rng_));

          costs.push_back(result->get_cost());

          std::unique_ptr<typename ModelT::Gradients> gradients(
              model_->compute_gradients(*result));

          if (check_gradients) {
              VLOG(1) << "Checking gradients.";

              // TODO(cvangysel): configure relative_error_threshold according to whether the
              //                  net has kinks or not.
              GradientCheckFn<typename ModelT::Wrapped>()(
                  model_.get(),
                  *batch,
                  *result,
                  *gradients,
                  1e-5 /* epsilon */,
                  1e-4 /* relative_error_threshold */,
                  rng_state,
                  &rng_);
          }

          if (backpropagate) {
              model_->update(*gradients, get_learning_rate(), result->scaled_regularization_lambda());
          }
      }

      return costs;
  }

  std::vector<FloatT> train_dummy_source(DataSourceType* const source) {
      setup_model(20, 3, 15, 4,
                  1, /* num_random_objects */
                  0.01 /* regularization_lambda */);
      setup_data(source);

      // Use smaller batch size when we have kinks.
      //
      // "One fix to the above problem of kinks is to use fewer datapoints, 
      //  since loss functions that contain kinks (e.g. due to use of ReLUs
      //  or margin losses etc.) will have fewer kinks with fewer datapoints,
      //  so it is less likely for you to cross one when you perform the finite
      //  difference approximation.
      //
      //  Moreover, if your gradcheck for only ~2 or 3 datapoints then you
      //  would almost certainly gradcheck for an entire batch.
      //  Using very few datapoints also makes your gradient check faster and more efficient."
      const size_t batch_size = 1024; // (
      //     get_desc().transform_desc().nonlinearity() ==
      //     lse::ModelDesc::TransformDesc::HARD_TANH) ? 8 : 1024;

      std::vector<FloatT> costs;

      for (size_t epoch = 0; epoch < 10; ++epoch) {
          const std::vector<FloatT> epoch_costs = train(
              batch_size,
              3, /* window_size */
              true /* check_gradients */);

          costs.insert(costs.end(),
                       epoch_costs.begin(), epoch_costs.end());

          data_->reset();
      }

      return costs;
  }

  std::unique_ptr<ModelT> model_;
  std::unique_ptr<DataSourceType> data_;

  RNG rng_;
};

#endif /* CUNVSM_TESTS_BASE_CUDA_H */