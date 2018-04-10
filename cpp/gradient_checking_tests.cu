#include "cuNVSM/tests_base_cuda.h"

#define COST_EPSILON 1e-2

using ::testing::Contains;

class TextEntityTest : public ModelTest<ModelTestWrapper<LSE>> {
 protected:
  virtual typename DataSourceType::BatchType* create_batch(
          const size_t batch_size,
          const size_t window_size) const {
      return new typename DataSourceType::BatchType(batch_size, window_size);
  }
};

class EntityEntityTest : public ModelTest<ModelTestWrapper<Model<EntityEntity::Objective>>> {
 protected:
  virtual typename DataSourceType::BatchType* create_batch(
          const size_t batch_size,
          const size_t window_size) const {
      return new typename DataSourceType::BatchType(batch_size);
  }
};

class TermTermTest : public ModelTest<ModelTestWrapper<Model<TermTerm::Objective>>> {
 protected:
  virtual typename DataSourceType::BatchType* create_batch(
          const size_t batch_size,
          const size_t window_size) const {
      return new typename DataSourceType::BatchType(batch_size);
  }
};

class TextEntityEntityEntityTest : public ModelTest<ModelTestWrapper<Model<TextEntityEntityEntity::Objective>>> {
 protected:
  virtual typename DataSourceType::BatchType* create_batch(
          const size_t batch_size,
          const size_t window_size) const {
      return new typename DataSourceType::BatchType(
          TextEntity::Batch(batch_size, window_size),
          EntityEntity::Batch(batch_size));
  }
};

class TextEntityTermTermTest : public ModelTest<ModelTestWrapper<Model<TextEntityTermTerm::Objective>>> {
 protected:
  virtual typename DataSourceType::BatchType* create_batch(
          const size_t batch_size,
          const size_t window_size) const {
      return new typename DataSourceType::BatchType(
          TextEntity::Batch(batch_size, window_size),
          TermTerm::Batch(batch_size));
  }
};

typedef TextEntityTest ConstantTextEntityTest;
typedef EntityEntityTest ConstantEntityEntityTest;
typedef TermTermTest ConstantTermTermTest;
typedef TextEntityEntityEntityTest ConstantTextEntityEntityEntityTest;
typedef TextEntityTermTermTest ConstantTextEntityTermTermTest;

// In the case that we feed the network constant input, we run a different configuration
// than for the "average" input or random input. This is because momentum-based update mechanisms
// quickly get out of whack when they "figure out" that it's always the same patterns.
//
// Also in the case of batch normalization, for a constant input source, batch normalization
// fails as in a single batch all rows are equal and consequently the activations become zero.
INSTANTIATE_TEST_CASE_P(RandomSeedAndConfigs,
                        ConstantTextEntityTest,
                        ::testing::Combine(
                            ::testing::Range<RNG::result_type>(0 /* start, inclusive */,
                                                               6 /* end, exclusive */,
                                                               1 /* step */),
                            ::testing::Values<std::string>(
                                // Different variants of Transform.
                                "transform_desc < "
                                "    batch_normalization: false "
                                "    nonlinearity: TANH "
                                ">",
                                "transform_desc < "
                                "    batch_normalization: false "
                                "    nonlinearity: HARD_TANH "
                                ">",

                                // Test bias_negative_samples.
                                "transform_desc < "
                                "    batch_normalization: false "
                                "    nonlinearity: TANH "  // Remove unnecessary kinks!
                                "> "
                                "bias_negative_samples: true",

                                // Test l2_normalize_reprs.
                                "transform_desc < "
                                "    batch_normalization: false "
                                "    nonlinearity: TANH "  // Remove unnecessary kinks!
                                "> "
                                "l2_normalize_phrase_reprs: true",
                                "transform_desc < "
                                "    batch_normalization: false "
                                "    nonlinearity: TANH "  // Remove unnecessary kinks!
                                "> "
                                "l2_normalize_entity_reprs: true ",
                                "transform_desc < "
                                "    batch_normalization: false "
                                "    nonlinearity: TANH "  // Remove unnecessary kinks!
                                "> "
                                "l2_normalize_phrase_reprs: true "
                                "l2_normalize_entity_reprs: true"
                            ),
                            ::testing::Values<std::string>(
                                "type: SGD")));

TEST_P(ConstantTextEntityTest, ConstantSource_GradientCheck) {
    auto costs = train_dummy_source(new ConstantSource);
    EXPECT_TRUE(!costs.empty());
}

INSTANTIATE_TEST_CASE_P(RandomSeedAndConfigs,
                        ConstantEntityEntityTest,
                        ::testing::Combine(
                            ::testing::Range<RNG::result_type>(0 /* start, inclusive */,
                                                               6 /* end, exclusive */,
                                                               1 /* step */),
                            ::testing::Values<std::string>(
                                // Only the default, as it doesn't matter for this objective.
                                "transform_desc < "
                                "    batch_normalization: false "
                                "    nonlinearity: TANH "
                                ">"
                            ),
                            ::testing::Values<std::string>(
                                "type: SGD",
                                "type: ADAGRAD")));

TEST_P(ConstantEntityEntityTest, ConstantSource_GradientCheck) {
    RNG rng;

    // We only have three entities in our model.
    std::vector<EntityEntity::InstanceT>* const data =
        new std::vector<EntityEntity::InstanceT>();

    while (data->size() < 3 * (1 << 10)) {
        data->push_back(std::make_tuple(0, 1, 1.0));
        data->push_back(std::make_tuple(1, 2, 0.5));
        data->push_back(std::make_tuple(2, 3, 1.0));
        data->push_back(std::make_tuple(0, 2, 1.0));
        data->push_back(std::make_tuple(1, 2, 1.0));
    }

    auto costs = train_dummy_source(new EntityEntity::DataSource(data, &rng));
    EXPECT_TRUE(!costs.empty());
}

INSTANTIATE_TEST_CASE_P(RandomSeedAndConfigs,
                        ConstantTermTermTest,
                        ::testing::Combine(
                            ::testing::Range<RNG::result_type>(0 /* start, inclusive */,
                                                               6 /* end, exclusive */,
                                                               1 /* step */),
                            ::testing::Values<std::string>(
                                // Only the default, as it doesn't matter for this objective.
                                "transform_desc < "
                                "    batch_normalization: false "
                                "    nonlinearity: TANH "
                                ">"
                            ),
                            ::testing::Values<std::string>(
                                "type: SGD",
                                "type: ADAGRAD")));

TEST_P(ConstantTermTermTest, ConstantSource_GradientCheck) {
    RNG rng;

    // There are 20 terms within the model.
    std::vector<TermTerm::InstanceT>* const data =
        new std::vector<TermTerm::InstanceT>();

    while (data->size() < 3 * (1 << 10)) {
        data->push_back(std::make_tuple(0, 19, 1.0));
        data->push_back(std::make_tuple(16, 2, 1.0));
        data->push_back(std::make_tuple(2, 13, 1.0));
        data->push_back(std::make_tuple(0, 2, 1.0));
        data->push_back(std::make_tuple(11, 12, 1.0));
    }

    auto costs = train_dummy_source(new TermTerm::DataSource(data, &rng));
    EXPECT_TRUE(!costs.empty());
}

INSTANTIATE_TEST_CASE_P(RandomSeedAndConfigs,
                        ConstantTextEntityEntityEntityTest,
                        ::testing::Combine(
                            ::testing::Range<RNG::result_type>(0 /* start, inclusive */,
                                                               6 /* end, exclusive */,
                                                               1 /* step */),
                            ::testing::Values<std::string>(
                                // Only the default, as it doesn't matter for this objective.
                                "transform_desc < "
                                "    batch_normalization: false "
                                "    nonlinearity: TANH "
                                ">"
                            ),
                            ::testing::Values<std::string>(
                                "type: SGD",
                                "type: ADAM "
                                "adam_conf: < mode: DENSE_UPDATE_DENSE_VARIANCE >")));

TEST_P(ConstantTextEntityEntityEntityTest, ConstantSource_GradientCheck) {
    RNG rng;

    // We only have three entities in our model.
    std::vector<EntityEntity::InstanceT>* const data =
        new std::vector<EntityEntity::InstanceT>();

    while (data->size() < 3 * (1 << 10)) {
        data->push_back(std::make_tuple(0, 1, 1.0));
        data->push_back(std::make_tuple(1, 2, 1.0));
        data->push_back(std::make_tuple(2, 3, 1.0));
        data->push_back(std::make_tuple(0, 2, 1.0));
        data->push_back(std::make_tuple(1, 2, 1.0));
    }

    auto source = new MultiSource<TextEntity::Batch, EntityEntity::Batch>(
        std::make_tuple<DataSource<TextEntity::Batch>*,
                        DataSource<EntityEntity::Batch>*>(
            new RandomSource(&rng),
            new EntityEntity::DataSource(data, &rng)));

    auto costs = train_dummy_source(source);
    EXPECT_TRUE(!costs.empty());
}

INSTANTIATE_TEST_CASE_P(RandomSeedAndConfigs,
                        ConstantTextEntityTermTermTest,
                        ::testing::Combine(
                            ::testing::Range<RNG::result_type>(0 /* start, inclusive */,
                                                               6 /* end, exclusive */,
                                                               1 /* step */),
                            ::testing::Values<std::string>(
                                // Only the default, as it doesn't matter for this objective.
                                "transform_desc < "
                                "    batch_normalization: false "
                                "    nonlinearity: TANH "
                                ">"
                            ),
                            ::testing::Values<std::string>(
                                "type: SGD",
                                "type: ADAM "
                                "adam_conf: < mode: DENSE_UPDATE_DENSE_VARIANCE >")));

TEST_P(ConstantTextEntityTermTermTest, ConstantSource_GradientCheck) {
    RNG rng;

    // There are 20 terms within the model.
    std::vector<EntityEntity::InstanceT>* const data =
        new std::vector<EntityEntity::InstanceT>();

    while (data->size() < 3 * (1 << 10)) {
        data->push_back(std::make_tuple(0, 19, 1.0));
        data->push_back(std::make_tuple(16, 2, 1.0));
        data->push_back(std::make_tuple(2, 13, 1.0));
        data->push_back(std::make_tuple(0, 2, 1.0));
        data->push_back(std::make_tuple(11, 12, 1.0));
    }

    auto source = new MultiSource<TextEntity::Batch, TermTerm::Batch>(
        std::make_tuple<DataSource<TextEntity::Batch>*,
                        DataSource<TermTerm::Batch>*>(
            new RandomSource(&rng),
            new TermTerm::DataSource(data, &rng)));

    auto costs = train_dummy_source(source);
    EXPECT_TRUE(!costs.empty());
}

INSTANTIATE_TEST_CASE_P(RandomSeedAndConfigs,
                        TextEntityTest,
                        ::testing::Combine(
                            ::testing::Range<RNG::result_type>(0 /* start, inclusive */,
                                                               6 /* end, exclusive */,
                                                               1 /* step */),
                            ::testing::Values<std::string>(
                                // Different variants of Transform.
                                "transform_desc < "
                                "    batch_normalization: false "
                                "    nonlinearity: TANH "
                                ">",
                                "transform_desc < "
                                "    batch_normalization: true "
                                "    nonlinearity: TANH "
                                ">",
                                "transform_desc < "
                                "    batch_normalization: false "
                                "    nonlinearity: HARD_TANH "
                                ">",
                                "transform_desc < "
                                "    batch_normalization: true "
                                "    nonlinearity: HARD_TANH "
                                ">",

                                // Test bias_negative_samples.
                                "transform_desc < "
                                "    batch_normalization: true "
                                "    nonlinearity: TANH "  // Remove unnecessary kinks!
                                "> "
                                "bias_negative_samples: true",

                                // Test l2_normalize_reprs.
                                "transform_desc < "
                                "    batch_normalization: true "
                                "    nonlinearity: TANH "  // Remove unnecessary kinks!
                                "> "
                                "l2_normalize_phrase_reprs: true",
                                "transform_desc < "
                                "    batch_normalization: true "
                                "    nonlinearity: TANH "  // Remove unnecessary kinks!
                                "> "
                                "l2_normalize_entity_reprs: true ",
                                "transform_desc < "
                                "    batch_normalization: true "
                                "    nonlinearity: TANH "  // Remove unnecessary kinks!
                                "> "
                                "l2_normalize_phrase_reprs: true "
                                "l2_normalize_entity_reprs: true"
                            ),
                            ::testing::Values<std::string>(
                                "type: SGD",
                                "type: ADAGRAD",
                                "type: ADAM "
                                "adam_conf: < mode: SPARSE >",
                                "type: ADAM "
                                "adam_conf: < mode: DENSE_UPDATE >",
                                "type: ADAM "
                                "adam_conf: < mode: DENSE_UPDATE_DENSE_VARIANCE >")));

TEST_P(TextEntityTest, RandomSource_GradientCheck) {
    train_dummy_source(new RandomSource(&rng_));
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}
