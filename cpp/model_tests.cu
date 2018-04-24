#include "cuNVSM/tests_base.h"
#include "cuNVSM/model.h"

using ::testing::DoubleEq;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::FloatEq;

class ParamsTest : public ::testing::TestWithParam<RNG::result_type> {
 protected:
  virtual void SetUp() override {
      rng_.seed(GetParam());
  }

  template <typename FloatT, typename IdxType>
  void initialize_range_representations(RepresentationsStorage<FloatT, IdxType>* const representations) {
      thrust::copy(thrust::make_counting_iterator(static_cast<size_t>(0)),
                   thrust::make_counting_iterator(representations->reprs_.size()),
                   begin(representations->reprs_));

      std::vector<float32> expected;
      for (size_t i = 0; i < representations->reprs_.size(); ++i) {
          expected.push_back(i);
      }

      ASSERT_THAT(to_host(representations->reprs_),
                  ElementsAreArray(expected));
  }

  void initialize_transform(Transform<FloatT>* const transform) {
      thrust::copy(thrust::make_counting_iterator(static_cast<size_t>(0)),
          thrust::make_counting_iterator(transform->transform_.size()),
          begin(transform->transform_));

      thrust::copy(thrust::make_counting_iterator(static_cast<size_t>(0)),
          thrust::make_counting_iterator(transform->bias_.size()),
          begin(transform->bias_));
      transform->bias_.scale(transform->bias_.getStream(), 1e-3);
  }

  RNG rng_;
};

INSTANTIATE_TEST_CASE_P(RandomSeed,
                        ParamsTest,
                        ::testing::Range<RNG::result_type>(0 /* start, inclusive */,
                                                           11 /* end, exclusive */,
                                                           1 /* step */));

TEST_P(ParamsTest, get_average_representations) {
    Representations<FloatT, DefaultModel::WordIdxType> representations(
        WORD_REPRS,
        4, /* num_objects */
        3, /* repr_size */
        ParseProto<UpdateMethodConf>("type: SGD"), /* update_method */
        DefaultStream::get());

    representations.initialize(&rng_);
    initialize_range_representations(&representations);

    std::unique_ptr<device_matrix<DefaultModel::WordIdxType>> indices(
        device_matrix<DefaultModel::WordIdxType>::create_column(
            DefaultStream::get()->next(),
            {1, 3, 2,
             0, 3, 1}));

    std::unique_ptr<device_matrix<FloatT>> averages(
        representations.get_average_representations(
            DefaultStream::get()->next(),
            *indices,
            3 /* window_size */));

    EXPECT_THAT(
        to_host(*averages),
        ElementsAre(FPHelper<FloatT>::eq((3 + 9  + 6) / 3.),
                    FPHelper<FloatT>::eq((4 + 10 + 7) / 3.),
                    FPHelper<FloatT>::eq((5 + 11 + 8) / 3.),
                    FPHelper<FloatT>::eq((0 + 9  + 3) / 3.),
                    FPHelper<FloatT>::eq((1 + 10 + 4) / 3.),
                    FPHelper<FloatT>::eq((2 + 11 + 5) / 3.)));
}

TEST_P(ParamsTest, get_weighted_average_representations) {
    Representations<FloatT, DefaultModel::WordIdxType> representations(
        WORD_REPRS,
        4, /* num_objects */
        3, /* repr_size */
        ParseProto<UpdateMethodConf>("type: SGD"), /* update_method */
        DefaultStream::get());

    representations.initialize(&rng_);
    initialize_range_representations(&representations);

    std::unique_ptr<device_matrix<DefaultModel::WordIdxType>> indices(
        device_matrix<DefaultModel::WordIdxType>::create_column(
            DefaultStream::get()->next(),
            {1, 3, 2,
             0, 3, 1}));

    std::unique_ptr<device_matrix<DefaultModel::FloatT>> indices_weights(
        device_matrix<DefaultModel::FloatT>::create_column(
            DefaultStream::get()->next(),
            {0.5, 0.3, 0.1,
             1.0, 2.0, 0.2}));

    std::unique_ptr<device_matrix<FloatT>> averages(
        representations.get_average_representations(
            DefaultStream::get()->next(),
            *indices,
            3, /* window_size */
            indices_weights.get()));

    EXPECT_THAT(
        to_host(*averages),
        ElementsAre(FPHelper<FloatT>::eq((0.5 * 3 + 0.3 * 9  + 0.1 * 6) / 3.),
                    FPHelper<FloatT>::eq((0.5 * 4 + 0.3 * 10 + 0.1 * 7) / 3.),
                    FPHelper<FloatT>::eq((0.5 * 5 + 0.3 * 11 + 0.1 * 8) / 3.),
                    FPHelper<FloatT>::eq((1.0 * 0 + 2.0 * 9  + 0.2 * 3) / 3.),
                    FPHelper<FloatT>::eq((1.0 * 1 + 2.0 * 10 + 0.2 * 4) / 3.),
                    FPHelper<FloatT>::eq((1.0 * 2 + 2.0 * 11 + 0.2 * 5) / 3.)));
}

TEST_P(ParamsTest, generate_labels) {
    const size_t num_words = 100;
    const size_t num_objects = 5000;

    DefaultModel lse(num_words, num_objects,
            ParseProto<lse::ModelDesc>("word_repr_size: 64 entity_repr_size: 32"),
            ParseProto<lse::TrainConfig>("update_method: < type: SGD >"));

    lse.initialize(&rng_);

    const size_t num_labels = 5;
    const size_t num_negative_labels = 10;

    std::vector<DefaultModel::EntityIdxType> labels = {1, 2, 3, 4, 5};

    std::vector<DefaultModel::EntityIdxType> instance_entities;

    lse.objective_->generate_labels(
        labels.data(),
        num_labels,
        num_negative_labels,
        &instance_entities,
        &rng_);

    EXPECT_THAT(instance_entities.size(),
                Eq(num_labels * (num_negative_labels + 1)));
}

TEST_P(ParamsTest, Representations_update) {
    Representations<FloatT, DefaultModel::WordIdxType> representations(
        WORD_REPRS,
        4, /* num_objects */
        3, /* repr_size */
        ParseProto<UpdateMethodConf>("type: SGD"), /* update_method */
        DefaultStream::get());

    representations.initialize(&rng_);

    {
        initialize_range_representations(&representations);

        TextEntity::Objective::ForwardResultType result(
            device_matrix<WordIdxType>::create(
                DefaultStream::get()->next(), {0, 3, 1, 0}, 1, 4), /* words */
            device_matrix<FloatT>::create(
                DefaultStream::get()->next(), {1.0, 1.0, 1.0, 1.0}, 1, 4),  /* word_weights */
            device_matrix<WordIdxType>::create(
                DefaultStream::get()->next(), {0}, 1, 1), /* entities */
            2, /* window_size */
            1, /* num_random_entities */
            0.1 /* regularization_lambda */);

        SingleGradients<FloatT> gradients(&result);
        gradients.grad_phrase_reprs_.reset(
            new device_matrix<FloatT>(3 /* repr_size */,
                                      2, /* num_phrase_reprs */
                                      DefaultStream::get()->next()));
        gradients.grad_phrase_reprs_->fillwith(
            gradients.grad_phrase_reprs_->getStream(), 0.0);

        representations.update(gradients,
                               0.1, /* learning_rate */
                               result.scaled_regularization_lambda(),
                               DefaultStream::get());

        const FloatT scale_factor = 1.0 - (0.1 * 0.1) / 2.0;

        EXPECT_THAT(
            to_host(representations.reprs_),
            ElementsAreArray({0. * scale_factor, 1. * scale_factor, 2. * scale_factor,
                              3. * scale_factor, 4. * scale_factor, 5. * scale_factor,
                              6. * scale_factor, 7. * scale_factor, 8. * scale_factor,
                              9. * scale_factor, 10. * scale_factor, 11. * scale_factor}));
    }

    {
        initialize_range_representations(&representations);

        TextEntity::Objective::ForwardResultType result(
            device_matrix<WordIdxType>::create(DefaultStream::get()->next(), {0, 3, 1, 0}, 1, 4), /* words */
            device_matrix<FloatT>::create(DefaultStream::get()->next(), {1.0, 1.0, 1.0, 1.0}, 1, 4), /* word_weights */
            device_matrix<WordIdxType>::create(DefaultStream::get()->next(), {0}, 1, 1), /* entities */
            2, /* window_size */
            1, /* num_random_entities */
            0.0 /* regularization_lambda */);

        SingleGradients<FloatT> gradients(&result);
        gradients.grad_phrase_reprs_.reset(
            new device_matrix<FloatT>(3, /* repr_size */
                                      2, /* num_phrase_reprs */
                                      DefaultStream::get()->next()));

        to_device({5.0, 4.0, 3.0,
                   -3.0, -2.0, 10.0},
                  gradients.grad_phrase_reprs_.get());

        representations.update(gradients,
                               0.1, /* learning_rate */
                               result.scaled_regularization_lambda(),
                               DefaultStream::get());

        const FloatT learning_rate = 0.1;

        EXPECT_THAT(
            to_host(representations.reprs_),
            ElementsAreArray({FPHelper<FloatT>::eq(0. + (5.0 + (-3.0)) * learning_rate),
                              FPHelper<FloatT>::eq(1. + (4.0 + (-2.0)) * learning_rate),
                              FPHelper<FloatT>::eq(2. + (3.0 + 10.0) *learning_rate),
                              FPHelper<FloatT>::eq(3. + (-3.0) * learning_rate),
                              FPHelper<FloatT>::eq(4. + (-2.0) * learning_rate),
                              FPHelper<FloatT>::eq(5. + 10.0 * learning_rate),
                              FPHelper<FloatT>::eq(6.),
                              FPHelper<FloatT>::eq(7.),
                              FPHelper<FloatT>::eq(8.),
                              FPHelper<FloatT>::eq(9. + (5.0) * learning_rate),
                              FPHelper<FloatT>::eq(10. + (4.0) * learning_rate),
                              FPHelper<FloatT>::eq(11. + 3.0 * learning_rate)}));
    }
}

TEST_P(ParamsTest, RepresentationsStorage_update_dense) {
    RepresentationsStorage<FloatT, DefaultModel::WordIdxType> representations(
        4, /* num_objects */
        3, /* repr_size */
        DefaultStream::get());

    initialize_range_representations(&representations);

    representations.update_dense(
        0, /* stream */
        thrust::make_constant_iterator(10.0),
        0.1, /* learning_rate */
        0.01 /* scaled_regularization_lambda */);

    EXPECT_THAT(
        to_host(representations.reprs_),
        ElementsAreArray({
            FPHelper<FloatT>::eq(0. * (1.0 - 0.01 * 0.1) + 10.0 * 0.1),
            FPHelper<FloatT>::eq(1. * (1.0 - 0.01 * 0.1) + 10.0 * 0.1),
            FPHelper<FloatT>::eq(2. * (1.0 - 0.01 * 0.1) + 10.0 * 0.1),
            FPHelper<FloatT>::eq(3. * (1.0 - 0.01 * 0.1) + 10.0 * 0.1),
            FPHelper<FloatT>::eq(4. * (1.0 - 0.01 * 0.1) + 10.0 * 0.1),
            FPHelper<FloatT>::eq(5. * (1.0 - 0.01 * 0.1) + 10.0 * 0.1),
            FPHelper<FloatT>::eq(6. * (1.0 - 0.01 * 0.1) + 10.0 * 0.1),
            FPHelper<FloatT>::eq(7. * (1.0 - 0.01 * 0.1) + 10.0 * 0.1),
            FPHelper<FloatT>::eq(8. * (1.0 - 0.01 * 0.1) + 10.0 * 0.1),
            FPHelper<FloatT>::eq(9. * (1.0 - 0.01 * 0.1) + 10.0 * 0.1),
            FPHelper<FloatT>::eq(10. * (1.0 - 0.01 * 0.1) + 10.0 * 0.1),
            FPHelper<FloatT>::eq(11. * (1.0 - 0.01 * 0.1) + 10.0 * 0.1),
        }));
}

TEST_P(ParamsTest, Transform) {
    lse::ModelDesc::TransformDesc desc;

    Transform<FloatT> transform(
        TRANSFORM,
        desc,
        3 /* word_repr_size */, 5 /* entity_repr_size */,
        ParseProto<UpdateMethodConf>("type: SGD"), /* update_method */
        DefaultStream::get());

    transform.initialize(&rng_);
    initialize_transform(&transform);

    /*
        Transform will be a 3-by-5 matrix:
            |  0  1  2  3  4 |
            |  5  6  7  8  9 |
            | 10 11 12 13 14 |

        Bias will be a 3-dimensional vector:
            | 0 1 2 3 4 | * 1e-3
    */

    device_matrix<FloatT> input_reprs(3, 2, NULL /* stream */);
    to_device({0.01, 0.02, 0.03,
               0.001, 0.002, 0.003}, &input_reprs);

    std::unique_ptr<device_matrix<FloatT>> result(
        transform.transform(
            DefaultStream::get()->next(),
            input_reprs,
            nullptr /* batch_normalization */));

    /*
        The first representation will undergo the following transition:
                  | 0    + 0.10 + 0.30 |     | 0.000 |           | 0.400 |
                  | 0.01 + 0.12 + 0.33 |     | 0.001 |           | 0.461 |
            tanh( | 0.02 + 0.14 + 0.36 |  +  | 0.002 | ) = tanh( | 0.522 | )
                  | 0.03 + 0.16 + 0.39 |     | 0.003 |           | 0.583 |
                  | 0.04 + 0.18 + 0.42 |     | 0.004 |           | 0.644 |

        For the second representation:
                  | 0     + 0.010 + 0.030 |     | 0.000 |           | 0.040 |
                  | 0.001 + 0.012 + 0.033 |     | 0.001 |           | 0.047 |
            tanh( | 0.002 + 0.014 + 0.036 |  +  | 0.002 | ) = tanh( | 0.054 | )
                  | 0.003 + 0.016 + 0.039 |     | 0.003 |           | 0.061 |
                  | 0.004 + 0.018 + 0.042 |     | 0.004 |           | 0.068 |
    */

    EXPECT_THAT(
        to_host(*result),
        ElementsAreArray({FPHelper<FloatT>::eq(tanh(0.400)),
                          FPHelper<FloatT>::eq(tanh(0.461)),
                          FPHelper<FloatT>::eq(tanh(0.522)),
                          FPHelper<FloatT>::eq(tanh(0.583)),
                          FPHelper<FloatT>::eq(tanh(0.644)),

                          FPHelper<FloatT>::eq(tanh(0.040)),
                          FPHelper<FloatT>::eq(tanh(0.047)),
                          FPHelper<FloatT>::eq(tanh(0.054)),
                          FPHelper<FloatT>::eq(tanh(0.061)),
                          FPHelper<FloatT>::eq(tanh(0.068))}));
}

TEST_P(ParamsTest, Transform_backward) {
    rng_.seed(10);

    lse::ModelDesc model_desc;
    model_desc.set_word_repr_size(2);
    model_desc.set_entity_repr_size(3);

    model_desc.set_bias_negative_samples(true);

    DefaultModel lse(5, /* num_words */
            3, /* num_objects */
            model_desc,
            ParseProto<lse::TrainConfig>(
                "num_random_entities: 10 "
                "regularization_lambda: 0.01 "
                "update_method: < type: SGD >") /* update_method */);
    lse.initialize(&rng_);

    TextEntity::Batch batch(32, /* batch_size */ 2 /* window_size */);

    ConstantSource data(
        2, /* feature_value */
        1 /* label */);
    data.next(&batch);

    std::unique_ptr<TextEntity::Objective::ForwardResultType> result(
        dynamic_cast<TextEntity::Objective::ForwardResultType*>(
            lse.compute_cost(batch, &rng_)));

    result->get_cost();

    std::unique_ptr<TextEntity::Objective::GradientsType> gradients(
        dynamic_cast<TextEntity::Objective::GradientsType*>(
            lse.compute_gradients(*result)));

    // dC / d W
    EXPECT_THAT(
        to_host(*gradients->grad_transform_matrix_),
        ElementsAreArray({
            FPHelper<FloatT>::eq(0.19053905536266260712),
            FPHelper<FloatT>::eq(-0.40135414958704901389),
            FPHelper<FloatT>::eq(-0.1114867197947356503),
            FPHelper<FloatT>::eq(0.34321179097285392512),
            FPHelper<FloatT>::eq(-0.72294614997420003633),
            FPHelper<FloatT>::eq(-0.20081739514038740579)
        }));

    // dC / d bias
    EXPECT_THAT(
        to_host(*gradients->grad_bias_),
        ElementsAreArray({
            FPHelper<FloatT>::eq(-0.59713496418425149326),
            FPHelper<FloatT>::eq(1.2578134980395563325),
            FPHelper<FloatT>::eq(0.34939093355395389739),
        }));

    // dC / d x
    EXPECT_THAT(
        to_host(*gradients->grad_phrase_reprs_),
        ElementsAreArray({
            FPHelper<FloatT>::eq(0.005498121774796882813),
            FPHelper<FloatT>::eq(-0.010496720853248608235),
            FPHelper<FloatT>::eq(0.0012117255393835533149),
            FPHelper<FloatT>::eq(-0.0093082500327885901031),
            FPHelper<FloatT>::eq(-0.016824653822594095448),
            FPHelper<FloatT>::eq(-0.010994870322164012819),
            FPHelper<FloatT>::eq(-0.018606242663242754387),
            FPHelper<FloatT>::eq(-0.023875877464595022387),
            FPHelper<FloatT>::eq(-0.027179035134069419455),
            FPHelper<FloatT>::eq(-0.021498935823674982654),
            FPHelper<FloatT>::eq(-0.0082518613517674373192),
            FPHelper<FloatT>::eq(-0.013371811963084052552),
            FPHelper<FloatT>::eq(-0.018606242663242754387),
            FPHelper<FloatT>::eq(-0.023875877464595022387),
            FPHelper<FloatT>::eq(-0.012538257587180764649),
            FPHelper<FloatT>::eq(-0.012183341142624032685),
            FPHelper<FloatT>::eq(-0.0082518613517674373192),
            FPHelper<FloatT>::eq(-0.013371811963084052552),
            FPHelper<FloatT>::eq(-0.014319846427829425323),
            FPHelper<FloatT>::eq(-0.025064348285055045723),
            FPHelper<FloatT>::eq(-0.012538257587180766384),
            FPHelper<FloatT>::eq(-0.012183341142624032685),
            FPHelper<FloatT>::eq(-0.011647463166856438649),
            FPHelper<FloatT>::eq(-0.0057428375714085287684),
            FPHelper<FloatT>::eq(0.0012117255393835541823),
            FPHelper<FloatT>::eq(-0.0093082500327885883684),
            FPHelper<FloatT>::eq(-0.014319846427829425323),
            FPHelper<FloatT>::eq(-0.025064348285055045723),
            FPHelper<FloatT>::eq(-0.013429052007505097588),
            FPHelper<FloatT>::eq(-0.018623844713839536602),
            FPHelper<FloatT>::eq(0.015852503086272205085),
            FPHelper<FloatT>::eq(7.3446482623583761478e-06),
            FPHelper<FloatT>::eq(-0.0091426557720917667887),
            FPHelper<FloatT>::eq(-0.019812315534299559938),
            FPHelper<FloatT>::eq(0.0012117255393835528812),
            FPHelper<FloatT>::eq(-0.0093082500327885883684),
            FPHelper<FloatT>::eq(-0.022892638898656086921),
            FPHelper<FloatT>::eq(-0.022687406644134999051),
            FPHelper<FloatT>::eq(-0.0048562595366784359896),
            FPHelper<FloatT>::eq(-0.021000786354759579805),
            FPHelper<FloatT>::eq(-0.016824653822594088509),
            FPHelper<FloatT>::eq(-0.010994870322164011084),
            FPHelper<FloatT>::eq(-0.013429052007505095853),
            FPHelper<FloatT>::eq(-0.018623844713839536602),
            FPHelper<FloatT>::eq(-0.019497037083567085591),
            FPHelper<FloatT>::eq(-0.030316381035810528038),
            FPHelper<FloatT>::eq(-0.02289263889865609039),
            FPHelper<FloatT>::eq(-0.022687406644134999051),
            FPHelper<FloatT>::eq(-0.027179035134069419455),
            FPHelper<FloatT>::eq(-0.021498935823674975715),
            FPHelper<FloatT>::eq(-0.040929018260633741322),
            FPHelper<FloatT>::eq(-0.024374026933510421766),
            FPHelper<FloatT>::eq(-0.0082518613517674373192),
            FPHelper<FloatT>::eq(-0.013371811963084050817),
            FPHelper<FloatT>::eq(-0.012538257587180764649),
            FPHelper<FloatT>::eq(-0.01218334114262403442),
            FPHelper<FloatT>::eq(-0.018606242663242757857),
            FPHelper<FloatT>::eq(-0.023875877464595022387),
            FPHelper<FloatT>::eq(0.010675312430534543082),
            FPHelper<FloatT>::eq(-0.0052446881024931241849),
            FPHelper<FloatT>::eq(-0.022892638898656079982),
            FPHelper<FloatT>::eq(-0.022687406644134999051),
            FPHelper<FloatT>::eq(-0.036642622025220401849),
            FPHelper<FloatT>::eq(-0.025562497753970448572)
        }));
}

TEST_P(ParamsTest, Transform_BatchNormalization) {
    TextEntity::Objective::ForwardResultType result;

    result.batch_normalization_.reset(new BatchNormalization<FloatT>(
        5, /* num_features */
        0.1, /* momentum */
        1e-5, /* epsilon */
        true /* cache_input */));

    lse::ModelDesc::TransformDesc desc;

    Transform<FloatT> transform(
        TRANSFORM,
        desc,
        3 /* word_repr_size */, 5 /* entity_repr_size */,
        ParseProto<UpdateMethodConf>("type: SGD"), /* update_method */
        DefaultStream::get());

    transform.initialize(&rng_);
    initialize_transform(&transform);

    /*
        Transform will be a 3-by-5 matrix:
            |  0  1  2  3  4 |
            |  5  6  7  8  9 |
            | 10 11 12 13 14 |

        Bias will be a 3-dimensional vector:
            | 0 1 2 3 4 | * 1e-3
    */

    device_matrix<FloatT> input_reprs(3, 2, NULL /* stream */);
    to_device({0.01, 0.02, 0.03,
               0.001, 0.002, 0.003}, &input_reprs);

    std::unique_ptr<device_matrix<FloatT>> output(
        transform.transform(
            DefaultStream::get()->next(),
            input_reprs,
            result.batch_normalization_.get() /* batch_normalization */));

    EXPECT_THAT(
        to_host(*output),
        ElementsAreArray({FPHelper<FloatT>::eq(0.7615293524851600715),
                          FPHelper<FloatT>::eq(0.76196488305628828908),
                          FPHelper<FloatT>::eq(0.76239459573842593976),
                          FPHelper<FloatT>::eq(0.76282051982955900726),
                          FPHelper<FloatT>::eq(0.76324378068549525445),

                          FPHelper<FloatT>::eq(-0.76152935248515996047),
                          FPHelper<FloatT>::eq(-0.76112478489184165475),
                          FPHelper<FloatT>::eq(-0.76071446308133705561),
                          FPHelper<FloatT>::eq(-0.76030038648736808504),
                          FPHelper<FloatT>::eq(-0.75988366421371911219)}));

    SingleGradients<FloatT> gradients(&result);
    gradients.grad_transform_matrix_.reset(new device_matrix<FloatT>(
        5, 3, DefaultStream::get()->next()));
    gradients.grad_bias_.reset(new device_matrix<FloatT>(
        5, 1, DefaultStream::get()->next()));

    device_matrix<FloatT> grad_output(5, 2, DefaultStream::get()->next());
    to_device({0.1, 0.1, 0.1, 0.1, 0.1,
               0.2, 0.2, 0.2, 0.2, 0.2}, &grad_output);

    transform.backward(
        DefaultStream::get()->next(),
        result,
        input_reprs,
        *output,
        &grad_output,
        &gradients);

    EXPECT_THAT(
        to_host_isfinite(*gradients.grad_transform_matrix_),
        Each(Eq(true)));

    EXPECT_THAT(
        to_host_isfinite(*gradients.grad_bias_),
        Each(Eq(true)));
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}
