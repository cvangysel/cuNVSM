#include "cuNVSM/tests_base.h"
#include "cuNVSM/updates.h"

using ::testing::ElementsAreArray;

class UpdatesTest : public ::testing::TestWithParam<::testing::tuple<FloatT, FloatT>> {
 protected:
  template <typename T>
  T* create_storage(const FloatT value,
                    const size_t first_size,
                    const size_t second_size,
                    Streams* const streams) const {
      T* const storage = new T(first_size, second_size, streams);
      dynamic_cast<Storage<FloatT>*>(storage)->initialize_with_constant(value);

      return storage;
  }

  FloatT scaled_regularization_lambda() const {
      return std::get<0>(GetParam());
  }

  FloatT learning_rate() const {
      return std::get<1>(GetParam());
  }
};

INSTANTIATE_TEST_CASE_P(Regularization,
                        UpdatesTest,
                        ::testing::Combine(
                            ::testing::Values<FloatT>(0.0, 0.1),
                            ::testing::Values<FloatT>(1.0, 0.5)));

TEST_P(UpdatesTest, SGDTransformGradientUpdater) {
    std::unique_ptr<TransformStorage<FloatT>> storage(
        create_storage<TransformStorage<FloatT>>(
            5.0, /* initial value */
            8, 3, DefaultStream::get()));

    SGDTransformGradientUpdater<FloatT> updater;

    device_matrix<FloatT> grad_matrix(3, 8, NULL /* stream */);
    to_device({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
               9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
               17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0}, &grad_matrix);

    device_matrix<FloatT> grad_bias(3, 1, NULL /* stream */);
    to_device({25.0, 26.0, 27.0}, &grad_bias);

    TransformStorage<FloatT>::GradientType gradient_desc = std::forward_as_tuple(
        grad_matrix, grad_bias);

    updater.update(storage.get(), &gradient_desc,
                   learning_rate(),
                   scaled_regularization_lambda(),
                   DefaultStream::get());

    EXPECT_THAT(
        to_host(*std::get<0>(storage->get())),
        ElementsAreArray({
            5.0 + learning_rate() * (1.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (2.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (3.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (4.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (5.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (6.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (7.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (8.0 - scaled_regularization_lambda() * 5.0),

            5.0 + learning_rate() * (9.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (10.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (11.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (12.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (13.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (14.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (15.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (16.0 - scaled_regularization_lambda() * 5.0),

            5.0 + learning_rate() * (17.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (18.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (19.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (20.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (21.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (22.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (23.0 - scaled_regularization_lambda() * 5.0),
            5.0 + learning_rate() * (24.0 - scaled_regularization_lambda() * 5.0)
    }));


    EXPECT_THAT(
        to_host(*std::get<1>(storage->get())),
        ElementsAreArray({
            5.0 + learning_rate() * 25.0,
            5.0 + learning_rate() * 26.0,
            5.0 + learning_rate() * 27.0
    }));
}

TEST_P(UpdatesTest, SGDRepresentationsGradientUpdater) {
    const size_t repr_size = 4;

    std::unique_ptr<RepresentationsStorage<FloatT, IdxType>> storage(
        create_storage<RepresentationsStorage<FloatT, IdxType>>(
            5.0, /* initial value */
            10, repr_size, DefaultStream::get()));

    SGDRepresentationsGradientUpdater<FloatT, IdxType> updater;

    device_matrix<FloatT> first_grad_repr(repr_size, 1, NULL /* stream */);
    to_device({2.0, 2.5, 3.0, 4.0},  // identity: (2.0 + 2.5 + 3.0 + 4.0) / 4 = 2.875
                                     // squared: (4.0 + 6.25 + 9.0 + 16.0) / 4 = 35.25 / 4 = 8.8125
              &first_grad_repr);

    device_matrix<FloatT> second_grad_repr(repr_size, 1, NULL /* stream */);
    to_device({10.0, 11.0, 12.0, 13.0},  // identity: (10.0 + 11.0 + 12.0 + 13.0) / 4 = 11.5
                                         // squared: (100.0 + 121.0 + 144.0 + 169.0) / 4 = 534.0 / 4 = 133.5
              &second_grad_repr);

    device_matrix<IdxType> first_repr_idx(1, 3, NULL /* stream */);
    to_device({9, 0, 1},
              &first_repr_idx);

    device_matrix<IdxType> second_repr_idx(1, 3, NULL /* stream */);
    to_device({5, 1, 8},
              &second_repr_idx);

    const size_t window_size = 3;

    RepresentationsStorage<FloatT, IdxType>::GradientType gradient_desc = {
        std::forward_as_tuple(first_grad_repr, first_repr_idx, window_size, nullptr),
        std::forward_as_tuple(second_grad_repr, second_repr_idx, window_size, nullptr),
    };

    updater.update(storage.get(), &gradient_desc,
                   learning_rate(),
                   scaled_regularization_lambda(),
                   DefaultStream::get());

    EXPECT_THAT(
        to_host(*storage->get()),
        ElementsAreArray({5.0 + learning_rate() * (2.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (2.5 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (3.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (4.0 - scaled_regularization_lambda() * 5.0),

                          5.0 + learning_rate() * (2.0 + 10.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (2.5 + 11.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (3.0 + 12.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (4.0 + 13.0 - scaled_regularization_lambda() * 5.0),

                          (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0,
                          (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0,
                          (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0,

                          5.0 + learning_rate() * (10.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (11.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (12.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (13.0 - scaled_regularization_lambda() * 5.0),

                          (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0,
                          (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0, (1.0 - learning_rate() * scaled_regularization_lambda()) * 5.0,

                          5.0 + learning_rate() * (10.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (11.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (12.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (13.0 - scaled_regularization_lambda() * 5.0),

                          5.0 + learning_rate() * (2.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (2.5 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (3.0 - scaled_regularization_lambda() * 5.0),
                          5.0 + learning_rate() * (4.0 - scaled_regularization_lambda() * 5.0)}));
}

TEST_P(UpdatesTest, AdagradTransformGradientUpdater) {
    const FloatT epsilon = 1e-6;

    std::unique_ptr<TransformStorage<FloatT>> storage(
        create_storage<TransformStorage<FloatT>>(
          5.0, /* initial value */
          8, 3, DefaultStream::get()));

    AdagradTransformGradientUpdater<FloatT> updater(
        8, /* source_vector_dim */
        3, /* dest_vector_dim */
        DefaultStream::get(),
        epsilon);

    device_matrix<FloatT> grad_matrix(3, 8, NULL /* stream */);
    to_device({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
               9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
               17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0}, &grad_matrix);

    device_matrix<FloatT> grad_bias(3, 1, NULL /* stream */);
    to_device({25.0, 26.0, 27.0}, &grad_bias);

    TransformStorage<FloatT>::GradientType gradient_desc = std::forward_as_tuple(
        grad_matrix, grad_bias);

    updater.update(storage.get(), &gradient_desc,
                   learning_rate(),
                   scaled_regularization_lambda(),
                   DefaultStream::get());

    EXPECT_THAT(
        to_host(*updater.storages_[0]->get_data()["transform"]),
        ElementsAreArray({1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0,
                          81.0, 100.0, 121.0, 144.0, 169.0, 196.0, 225.0, 256.0,
                          289.0, 324.0, 361.0, 400.0, 441.0, 484.0, 529.0, 576.0}));

    EXPECT_THAT(
        to_host(*updater.storages_[0]->get_data()["bias"]),
        ElementsAreArray({625.0,
                          676.0,
                          729.0}));

    EXPECT_THAT(
        to_host(grad_matrix),
        ElementsAreArray({FPHelper<FloatT>::eq(1.0 / sqrt(1.0 + epsilon)),
                          FPHelper<FloatT>::eq(2.0 / sqrt(4.0 + epsilon)),
                          FPHelper<FloatT>::eq(3.0 / sqrt(9.0 + epsilon)),
                          FPHelper<FloatT>::eq(4.0 / sqrt(16.0 + epsilon)),
                          FPHelper<FloatT>::eq(5.0 / sqrt(25.0 + epsilon)),
                          FPHelper<FloatT>::eq(6.0 / sqrt(36.0 + epsilon)),
                          FPHelper<FloatT>::eq(7.0 / sqrt(49.0 + epsilon)),
                          FPHelper<FloatT>::eq(8.0 / sqrt(64.0 + epsilon)),
                          FPHelper<FloatT>::eq(9.0 / sqrt(81.0 + epsilon)),
                          FPHelper<FloatT>::eq(10.0 / sqrt(100.0 + epsilon)),
                          FPHelper<FloatT>::eq(11.0 / sqrt(121.0 + epsilon)),
                          FPHelper<FloatT>::eq(12.0 / sqrt(144.0 + epsilon)),
                          FPHelper<FloatT>::eq(13.0 / sqrt(169.0 + epsilon)),
                          FPHelper<FloatT>::eq(14.0 / sqrt(196.0 + epsilon)),
                          FPHelper<FloatT>::eq(15.0 / sqrt(225.0 + epsilon)),
                          FPHelper<FloatT>::eq(16.0 / sqrt(256.0 + epsilon)),
                          FPHelper<FloatT>::eq(17.0 / sqrt(289.0 + epsilon)),
                          FPHelper<FloatT>::eq(18.0 / sqrt(324.0 + epsilon)),
                          FPHelper<FloatT>::eq(19.0 / sqrt(361.0 + epsilon)),
                          FPHelper<FloatT>::eq(20.0 / sqrt(400.0 + epsilon)),
                          FPHelper<FloatT>::eq(21.0 / sqrt(441.0 + epsilon)),
                          FPHelper<FloatT>::eq(22.0 / sqrt(484.0 + epsilon)),
                          FPHelper<FloatT>::eq(23.0 / sqrt(529.0 + epsilon)),
                          FPHelper<FloatT>::eq(24.0 / sqrt(576.0 + epsilon))}));

    EXPECT_THAT(
        to_host(grad_bias),
        ElementsAreArray({FPHelper<FloatT>::eq(25.0 / sqrt(625.0 + epsilon)),
                          FPHelper<FloatT>::eq(26.0 / sqrt(676.0 + epsilon)),
                          FPHelper<FloatT>::eq(27.0 / sqrt(729.0 + epsilon))}));
}

TEST_P(UpdatesTest, AdagradRepresentationsGradientUpdater) {
    const FloatT epsilon = 1e-6;

    std::unique_ptr<RepresentationsStorage<FloatT, IdxType>> storage(
        create_storage<RepresentationsStorage<FloatT, IdxType>>(
            5.0, /* initial value */
            10, 4, DefaultStream::get()));

    AdagradRepresentationsGradientUpdater<FloatT, IdxType> updater(
        10, /* num_objects */
        DefaultStream::get(),
        epsilon);

    device_matrix<FloatT> grad_repr(4, 2, NULL /* stream */);
    to_device({2.0, 2.5, 3.0, 4.0,  // (4.0 + 6.25 + 9.0 + 16.0) / 4 = 35.25 / 4 = 8.8125
               10.0, 11.0, 12.0, 13.0},  // (100.0 + 121.0 + 144.0 + 169.0)/ 4 = 534.0 / 4 = 133.5
              &grad_repr);

    device_matrix<IdxType> repr_idx(1, 6, NULL /* stream */);
    to_device({9, 0, 1, 5, 1, 8},
              &repr_idx);

    const size_t window_size = 3;

    RepresentationsStorage<FloatT, IdxType>::GradientType gradient_desc = {
        std::forward_as_tuple(grad_repr, repr_idx, window_size, nullptr)
    };

    updater.update(storage.get(), &gradient_desc,
                   learning_rate(),
                   scaled_regularization_lambda(),
                   DefaultStream::get());

    EXPECT_THAT(
        to_host(*updater.storages_[0]->get_data()["representations"]),
        ElementsAreArray({8.8125, 142.3125, 0.0, 0.0, 0.0, 133.5, 0.0, 0.0, 133.5, 8.8125}));

    EXPECT_THAT(
        to_host(grad_repr),
        ElementsAreArray({FPHelper<FloatT>::eq(2.0 / sqrt(((8.8125 + 8.8125 + 142.3125) / 3.0) + epsilon)),
                          FPHelper<FloatT>::eq(2.5 / sqrt(((8.8125 + 8.8125 + 142.3125) / 3.0) + epsilon)),
                          FPHelper<FloatT>::eq(3.0 / sqrt(((8.8125 + 8.8125 + 142.3125) / 3.0) + epsilon)),
                          FPHelper<FloatT>::eq(4.0 / sqrt(((8.8125 + 8.8125 + 142.3125) / 3.0) + epsilon)),
                          FPHelper<FloatT>::eq(10.0 / sqrt(((133.5 + 142.3125 + 133.5) / 3.0) + epsilon)),
                          FPHelper<FloatT>::eq(11.0 / sqrt(((133.5 + 142.3125 + 133.5) / 3.0) + epsilon)),
                          FPHelper<FloatT>::eq(12.0 / sqrt(((133.5 + 142.3125 + 133.5) / 3.0) + epsilon)),
                          FPHelper<FloatT>::eq(13.0 / sqrt(((133.5 + 142.3125 + 133.5) / 3.0) + epsilon))}));
}

TEST_P(UpdatesTest, AdamTransformGradientUpdater) {
    const FloatT epsilon = 1e-5;

    const FloatT beta1 = 0.9;
    const FloatT beta2 = 0.999;

    std::unique_ptr<TransformStorage<FloatT>> storage(
        create_storage<TransformStorage<FloatT>>(
            5.0, /* initial value */
            8, 3, DefaultStream::get()));

    AdamTransformGradientUpdater<FloatT> updater(
        8, /* source_vector_dim */
        3, /* dest_vector_dim */
        DefaultStream::get(),
        beta1,
        beta2,
        epsilon);

    {
        device_matrix<FloatT> grad_matrix(3, 8, NULL /* stream */);
        to_device({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                   9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                   17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0}, &grad_matrix);

        device_matrix<FloatT> grad_bias(3, 1, NULL /* stream */);
        to_device({25.0, 26.0, 27.0}, &grad_bias);

        TransformStorage<FloatT>::GradientType gradient_desc = std::forward_as_tuple(
            grad_matrix, grad_bias);

        updater.update(storage.get(), &gradient_desc,
                       learning_rate(),
                       scaled_regularization_lambda(),
                       DefaultStream::get());

        const FloatT bias_correction_t1 = sqrt(1.0 - pow(beta2, 1)) / (1.0 - pow(beta1, 1));

        for (size_t raw_g_idx = 0; raw_g_idx < 24; ++raw_g_idx) {
            const FloatT raw_g = static_cast<FloatT>(raw_g_idx + 1);
            const FloatT g = raw_g - scaled_regularization_lambda() * 5.0;

            EXPECT_THAT(
                to_host(grad_matrix)[raw_g_idx],
                FPHelper<FloatT>::eq(
                  bias_correction_t1 * 
                  ((1.0 - beta1) * g) /
                  (sqrt((1.0 - beta2) * pow(g, 2)) + epsilon))
            );
        }

        EXPECT_THAT(
            to_host(grad_bias),
            ElementsAreArray({FPHelper<FloatT>::eq(0.9999873510493572093),
                              FPHelper<FloatT>::eq(0.99998783754154196846),
                              FPHelper<FloatT>::eq(0.99998828799769046149)}));

        EXPECT_THAT(
            to_host(*updater.storages_[0]->get_data()["bias"]),
            ElementsAreArray({FPHelper<FloatT>::eq(2.5),
                              FPHelper<FloatT>::eq(2.6),
                              FPHelper<FloatT>::eq(2.7)}));

        EXPECT_THAT(
            to_host(*updater.storages_[1]->get_data()["bias"]),
            ElementsAreArray({FPHelper<FloatT>::eq(0.62500000000000055511),
                              FPHelper<FloatT>::eq(0.67600000000000060041),
                              FPHelper<FloatT>::eq(0.72900000000000064748)}));
    }

    {
        const std::vector<FloatT> param_transform_before_update =
            to_host(*std::get<0>(storage->get()));

        device_matrix<FloatT> grad_matrix(3, 8, NULL /* stream */);
        to_device({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                   9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                   17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0}, &grad_matrix);

        device_matrix<FloatT> grad_bias(3, 1, NULL /* stream */);
        to_device({25.0, 26.0, 27.0}, &grad_bias);

        TransformStorage<FloatT>::GradientType gradient_desc = std::forward_as_tuple(
            grad_matrix, grad_bias);

        updater.update(storage.get(), &gradient_desc,
                       learning_rate(),
                       scaled_regularization_lambda(),
                       DefaultStream::get());

        const FloatT bias_correction_t2 = sqrt(1.0 - pow(beta2, 2)) / (1.0 - pow(beta1, 2));

        for (size_t raw_g_idx = 0; raw_g_idx < 24; ++raw_g_idx) {
            const FloatT raw_g = static_cast<FloatT>(raw_g_idx + 1);
            const FloatT g = raw_g - scaled_regularization_lambda() * param_transform_before_update[raw_g_idx];

            const FloatT m_t1 = (1.0 - beta1) * (raw_g - scaled_regularization_lambda() * 5.0);
            const FloatT v_t1 = (1.0 - beta2) * pow(raw_g - scaled_regularization_lambda() * 5.0, 2);

            EXPECT_THAT(
                to_host(grad_matrix)[raw_g_idx],
                FPHelper<FloatT>::eq(
                  bias_correction_t2 * 
                  (beta1 * m_t1 + (1.0 - beta1) * g) /
                  (sqrt(beta2 * v_t1 + (1.0 - beta2) * pow(g, 2)) + epsilon))
            );
        }

        EXPECT_THAT(
            to_host(grad_bias),
            ElementsAreArray({FPHelper<FloatT>::eq(1.0523589755648365962),
                              FPHelper<FloatT>::eq(1.0523593375842164033),
                              FPHelper<FloatT>::eq(1.0523596727875677015)}));

        EXPECT_THAT(
            to_host(*updater.storages_[0]->get_data()["bias"]),
            ElementsAreArray({FPHelper<FloatT>::eq(4.9999999999999991118),
                              FPHelper<FloatT>::eq(5.1999999999999992895),
                              FPHelper<FloatT>::eq(5.3999999999999985789)}));

        EXPECT_THAT(
            to_host(*updater.storages_[1]->get_data()["bias"]),
            ElementsAreArray({FPHelper<FloatT>::eq(1.2500000000000011102),
                              FPHelper<FloatT>::eq(1.3520000000000012008),
                              FPHelper<FloatT>::eq(1.458000000000001295)}));
    }
}

TEST_P(UpdatesTest, AdamRepresentationsGradientUpdater_SPARSE) {
    const size_t repr_size = 4;

    const FloatT epsilon = 1e-5;

    const FloatT beta1 = 0.9;
    const FloatT beta2 = 0.999;

    std::unique_ptr<RepresentationsStorage<FloatT, IdxType>> storage(
        create_storage<RepresentationsStorage<FloatT, IdxType>>(
            5.0, /* initial value */
            5, 4, DefaultStream::get()));

    AdamRepresentationsGradientUpdater<FloatT, IdxType> updater(
        5, /* num_objects */
        repr_size,
        ParseProto<AdamConf>("mode: SPARSE"), /* conf */
        DefaultStream::get(),
        beta1,
        beta2,
        epsilon);

    device_matrix<FloatT> grad_repr(repr_size, 2, NULL /* stream */);
    to_device({2.0, 2.5, 3.0, 4.0,  // identity: (2.0 + 2.5 + 3.0 + 4.0) / 4 = 2.875
                                    // squared: (4.0 + 6.25 + 9.0 + 16.0) / 4 = 35.25 / 4 = 8.8125
               10.0, 11.0, 12.0, 13.0},  // identity: (10.0 + 11.0 + 12.0 + 13.0) / 4 = 11.5
                                         // squared: (100.0 + 121.0 + 144.0 + 169.0)/ 4 = 534.0 / 4 = 133.5
              &grad_repr);

    device_matrix<IdxType> repr_idx(1, 6, NULL /* stream */);
    to_device({4, 0, 1, 3, 1, 2},
              &repr_idx);

    const size_t window_size = 3;

    RepresentationsStorage<FloatT, IdxType>::GradientType gradient_desc = {
        std::forward_as_tuple(grad_repr, repr_idx, window_size, nullptr),
    };

    updater.update(storage.get(), &gradient_desc,
                   learning_rate(),
                   scaled_regularization_lambda(),
                   DefaultStream::get());

    EXPECT_THAT(
        to_host(*updater.storages_[0]->get_data()["representations"]),
        ElementsAreArray({(1.0 - beta1) * 2.0, (1.0 - beta1) * 2.5, (1.0 - beta1) * 3.0, (1.0 - beta1) * 4.0,
                          (1.0 - beta1) * (2.0 + 10.0), (1.0 - beta1) * (2.5 + 11.0), (1.0 - beta1) * (3.0 + 12.0), (1.0 - beta1) * (4.0 + 13.0),
                          (1.0 - beta1) * 10.0, (1.0 - beta1) * 11.0, (1.0 - beta1) * 12.0, (1.0 - beta1) * 13.0,
                          (1.0 - beta1) * 10.0, (1.0 - beta1) * 11.0, (1.0 - beta1) * 12.0, (1.0 - beta1) * 13.0,
                          (1.0 - beta1) * 2.0, (1.0 - beta1) * 2.5, (1.0 - beta1) * 3.0, (1.0 - beta1) * 4.0}));

    EXPECT_THAT(
        to_host(*updater.storages_[1]->get_data()["representations"]),
        ElementsAreArray({(1.0 - beta2) * 8.8125,
                          (1.0 - beta2) * (8.8125 + 133.5),
                          (1.0 - beta2) * 133.5,
                          (1.0 - beta2) * 133.5,
                          (1.0 - beta2) * 8.8125}));

    const FloatT bias_correction = sqrt(1.0 - pow(beta2, 1)) / (1.0 - pow(beta1, 1));

    EXPECT_THAT(
        to_host(grad_repr),
        ElementsAreArray({FPHelper<FloatT>::eq(
                              bias_correction *
                              (((1.0 - beta1) * 2.0 + (1.0 - beta1) * 2.0 + (1.0 - beta1) * (2.0 + 10.0)) / window_size) /
                              (sqrt(((1.0 - beta2) * 8.8125 + (1.0 - beta2) * 8.8125 + (1.0 - beta2) * (8.8125 + 133.5)) / window_size) + epsilon)),
                          FPHelper<FloatT>::eq(
                              bias_correction *
                              ((1.0 - beta1) * (2.5 + 2.5 + (2.5 + 11.0)) / window_size) /
                              (sqrt((1.0 - beta2) * (8.8125 + 8.8125 + 133.5 + 8.8125) / window_size) + epsilon)),
                          FPHelper<FloatT>::eq(
                              bias_correction *
                              ((1.0 - beta1) * (3.0 + 3.0 + (3.0 + 12.0)) / window_size) /
                              (sqrt((1.0 - beta2) * (8.8125 + 8.8125 + 133.5 + 8.8125) / window_size) + epsilon)),
                          FPHelper<FloatT>::eq(
                              bias_correction *
                              ((1.0 - beta1) * (4.0 + 4.0 + (4.0 + 13.0)) / window_size) /
                              (sqrt((1.0 - beta2) * (8.8125 + 8.8125 + 133.5 + 8.8125) / window_size) + epsilon)),

                          FPHelper<FloatT>::eq(
                              bias_correction *
                              ((1.0 - beta1) * (10.0 + (2.0 + 10.0) + 10.0) / window_size) /
                              (sqrt((1.0 - beta2) * (133.5 + (8.8125 + 133.5) + 133.5) / window_size) + epsilon)),
                          FPHelper<FloatT>::eq(
                              bias_correction *
                              ((1.0 - beta1) * (11.0 + (2.5 + 11.0) + 11.0) / window_size) /
                              (sqrt((1.0 - beta2) * (133.5 + (8.8125 + 133.5) + 133.5) / window_size) + epsilon)),
                          FPHelper<FloatT>::eq(
                              bias_correction *
                              ((1.0 - beta1) * (12.0 + (3.0 + 12.0) + 12.0) / window_size) /
                              (sqrt((1.0 - beta2) * (133.5 + (8.8125 + 133.5) + 133.5) / window_size) + epsilon)),
                          FPHelper<FloatT>::eq(
                              bias_correction *
                              ((1.0 - beta1) * (13.0 + (4.0 + 13.0) + 13.0) / window_size) /
                              (sqrt((1.0 - beta2) * (133.5 + (8.8125 + 133.5) + 133.5) / window_size) + epsilon)),
                        }));
}

TEST_P(UpdatesTest, AdamRepresentationsGradientUpdater_DENSE_UPDATE) {
    const size_t repr_size = 4;

    const FloatT epsilon = 1e-5;

    const FloatT beta1 = 0.9;
    const FloatT beta2 = 0.999;

    std::unique_ptr<RepresentationsStorage<FloatT, IdxType>> storage(
        create_storage<RepresentationsStorage<FloatT, IdxType>>(
            5.0, /* initial value */
            5, 4, DefaultStream::get()));

    AdamRepresentationsGradientUpdater<FloatT, IdxType> updater(
        5, /* num_objects */
        repr_size,
        ParseProto<AdamConf>("mode: DENSE_UPDATE"), /* conf */
        DefaultStream::get(),
        beta1,
        beta2,
        epsilon);

    device_matrix<FloatT> first_grad_repr(repr_size, 1, NULL /* stream */);
    to_device({2.0, 2.5, 3.0, 4.0},  // identity: (2.0 + 2.5 + 3.0 + 4.0) / 4 = 2.875
                                     // squared: (4.0 + 6.25 + 9.0 + 16.0) / 4 = 35.25 / 4 = 8.8125
              &first_grad_repr);

    device_matrix<FloatT> second_grad_repr(repr_size, 1, NULL /* stream */);
    to_device({10.0, 11.0, 12.0, 13.0},  // identity: (10.0 + 11.0 + 12.0 + 13.0) / 4 = 11.5
                                         // squared: (100.0 + 121.0 + 144.0 + 169.0) / 4 = 534.0 / 4 = 133.5
              &second_grad_repr);

    device_matrix<IdxType> first_repr_idx(1, 3, NULL /* stream */);
    to_device({4, 0, 1},
              &first_repr_idx);

    device_matrix<IdxType> second_repr_idx(1, 3, NULL /* stream */);
    to_device({3, 1, 2},
              &second_repr_idx);

    const size_t window_size = 3;

    RepresentationsStorage<FloatT, IdxType>::GradientType gradient_desc = {
        std::forward_as_tuple(first_grad_repr, first_repr_idx, window_size, nullptr),
        std::forward_as_tuple(second_grad_repr, second_repr_idx, window_size, nullptr),
    };

    EXPECT_THAT(
        to_host(*storage->get()),
        ElementsAreArray({5.0, 5.0, 5.0, 5.0,
                          5.0, 5.0, 5.0, 5.0,
                          5.0, 5.0, 5.0, 5.0,
                          5.0, 5.0, 5.0, 5.0,
                          5.0, 5.0, 5.0, 5.0}));

    updater.update(storage.get(), &gradient_desc,
                   learning_rate(),
                   scaled_regularization_lambda(),
                   DefaultStream::get());

    EXPECT_THAT(
        to_host(*updater.storages_[0]->get_data()["representations"]),
        ElementsAreArray({(1.0 - beta1) * 2.0, (1.0 - beta1) * 2.5, (1.0 - beta1) * 3.0, (1.0 - beta1) * 4.0,
                          (1.0 - beta1) * (2.0 + 10.0), (1.0 - beta1) * (2.5 + 11.0), (1.0 - beta1) * (3.0 + 12.0), (1.0 - beta1) * (4.0 + 13.0),
                          (1.0 - beta1) * 10.0, (1.0 - beta1) * 11.0, (1.0 - beta1) * 12.0, (1.0 - beta1) * 13.0,
                          (1.0 - beta1) * 10.0, (1.0 - beta1) * 11.0, (1.0 - beta1) * 12.0, (1.0 - beta1) * 13.0,
                          (1.0 - beta1) * 2.0, (1.0 - beta1) * 2.5, (1.0 - beta1) * 3.0, (1.0 - beta1) * 4.0}));

    EXPECT_THAT(
        to_host(*updater.storages_[1]->get_data()["representations"]),
        ElementsAreArray({(1.0 - beta2) * 8.8125,
                          (1.0 - beta2) * (8.8125 + 133.5),
                          (1.0 - beta2) * 133.5,
                          (1.0 - beta2) * 133.5,
                          (1.0 - beta2) * 8.8125}));

    const FloatT bias_correction = sqrt(1.0 - pow(beta2, 1)) / (1.0 - pow(beta1, 1));

    EXPECT_THAT(
        to_host(*storage->get()),
        ElementsAreArray({FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 2.0 / (sqrt((1.0 - beta2) * 8.8125) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 2.5 / (sqrt((1.0 - beta2) * 8.8125) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 3.0/ (sqrt((1.0 - beta2) * 8.8125) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 4.0 / (sqrt((1.0 - beta2) * 8.8125) + epsilon) - scaled_regularization_lambda() * 5.0)),

                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (2.0 + 10.0) / (sqrt((1.0 - beta2) * (8.8125 + 133.5)) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (2.5 + 11.0) / (sqrt((1.0 - beta2) * (8.8125 + 133.5)) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (3.0 + 12.0) / (sqrt((1.0 - beta2) * (8.8125 + 133.5)) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (4.0 + 13.0) / (sqrt((1.0 - beta2) * (8.8125 + 133.5)) + epsilon) - scaled_regularization_lambda() * 5.0)),

                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 10.0 / (sqrt((1.0 - beta2) * 133.5) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 11.0 / (sqrt((1.0 - beta2) * 133.5) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 12.0 / (sqrt((1.0 - beta2) * 133.5) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 13.0 / (sqrt((1.0 - beta2) * 133.5) + epsilon) - scaled_regularization_lambda() * 5.0)),

                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 10.0 / (sqrt((1.0 - beta2) * 133.5) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 11.0 / (sqrt((1.0 - beta2) * 133.5) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 12.0 / (sqrt((1.0 - beta2) * 133.5) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 13.0 / (sqrt((1.0 - beta2) * 133.5) + epsilon) - scaled_regularization_lambda() * 5.0)),

                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 2.0 / (sqrt((1.0 - beta2) * 8.8125) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 2.5 / (sqrt((1.0 - beta2) * 8.8125) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 3.0 / (sqrt((1.0 - beta2) * 8.8125) + epsilon) - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * 4.0 / (sqrt((1.0 - beta2) * 8.8125) + epsilon) - scaled_regularization_lambda() * 5.0))}));
}

TEST_P(UpdatesTest, AdamRepresentationsGradientUpdater_DENSE_UPDATE_DENSE_VARIANCE) {
    const size_t repr_size = 4;

    const FloatT epsilon = 1e-5;

    const FloatT beta1 = 0.9;
    const FloatT beta2 = 0.999;

    std::unique_ptr<RepresentationsStorage<FloatT, IdxType>> storage(
        create_storage<RepresentationsStorage<FloatT, IdxType>>(
            5.0, /* initial value */
            5, 4, DefaultStream::get()));

    AdamRepresentationsGradientUpdater<FloatT, IdxType> updater(
        5, /* num_objects */
        repr_size,
        ParseProto<AdamConf>("mode: DENSE_UPDATE_DENSE_VARIANCE"), /* conf */
        DefaultStream::get(),
        beta1,
        beta2,
        epsilon);

    device_matrix<FloatT> first_grad_repr(repr_size, 1, NULL /* stream */);
    to_device({2.0, 2.5, 3.0, 4.0},  // identity: (2.0 + 2.5 + 3.0 + 4.0) / 4 = 2.875
                                     // squared: (4.0 + 6.25 + 9.0 + 16.0) / 4 = 35.25 / 4 = 8.8125
              &first_grad_repr);

    device_matrix<FloatT> second_grad_repr(repr_size, 1, NULL /* stream */);
    to_device({10.0, 11.0, 12.0, 13.0},  // identity: (10.0 + 11.0 + 12.0 + 13.0) / 4 = 11.5
                                         // squared: (100.0 + 121.0 + 144.0 + 169.0) / 4 = 534.0 / 4 = 133.5
              &second_grad_repr);

    device_matrix<IdxType> first_repr_idx(1, 3, NULL /* stream */);
    to_device({4, 0, 1},
              &first_repr_idx);

    device_matrix<IdxType> second_repr_idx(1, 3, NULL /* stream */);
    to_device({3, 1, 2},
              &second_repr_idx);

    const size_t window_size = 3;

    RepresentationsStorage<FloatT, IdxType>::GradientType gradient_desc = {
        std::forward_as_tuple(first_grad_repr, first_repr_idx, window_size, nullptr),
        std::forward_as_tuple(second_grad_repr, second_repr_idx, window_size, nullptr),
    };

    EXPECT_THAT(
        to_host(*storage->get()),
        ElementsAreArray({5.0, 5.0, 5.0, 5.0,
                          5.0, 5.0, 5.0, 5.0,
                          5.0, 5.0, 5.0, 5.0,
                          5.0, 5.0, 5.0, 5.0,
                          5.0, 5.0, 5.0, 5.0}));

    updater.update(storage.get(), &gradient_desc,
                   learning_rate(),
                   scaled_regularization_lambda(),
                   DefaultStream::get());

    EXPECT_THAT(
        to_host(*updater.storages_[0]->get_data()["representations"]),
        ElementsAreArray({FPHelper<FloatT>::eq((1.0 - beta1) * (2.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (2.5 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (3.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (4.0 - scaled_regularization_lambda() * 5.0)),

                          FPHelper<FloatT>::eq((1.0 - beta1) * (2.0 + 10.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (2.5 + 11.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (3.0 + 12.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (4.0 + 13.0 - scaled_regularization_lambda() * 5.0)),

                          FPHelper<FloatT>::eq((1.0 - beta1) * (10.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (11.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (12.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (13.0 - scaled_regularization_lambda() * 5.0)),

                          FPHelper<FloatT>::eq((1.0 - beta1) * (10.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (11.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (12.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (13.0 - scaled_regularization_lambda() * 5.0)),

                          FPHelper<FloatT>::eq((1.0 - beta1) * (2.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (2.5 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (3.0 - scaled_regularization_lambda() * 5.0)),
                          FPHelper<FloatT>::eq((1.0 - beta1) * (4.0 - scaled_regularization_lambda() * 5.0))}));

    EXPECT_THAT(
        to_host(*updater.storages_[1]->get_data()["representations"]),
        ElementsAreArray({(1.0 - beta2) * pow(2.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(2.5 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(3.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(4.0 - scaled_regularization_lambda() * 5.0, 2),

                          (1.0 - beta2) * pow(2.0 + 10.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(2.5 + 11.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(3.0 + 12.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(4.0 + 13.0 - scaled_regularization_lambda() * 5.0, 2),

                          (1.0 - beta2) * pow(10.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(11.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(12.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(13.0 - scaled_regularization_lambda() * 5.0, 2),

                          (1.0 - beta2) * pow(10.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(11.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(12.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(13.0 - scaled_regularization_lambda() * 5.0, 2),

                          (1.0 - beta2) * pow(2.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(2.5 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(3.0 - scaled_regularization_lambda() * 5.0, 2),
                          (1.0 - beta2) * pow(4.0 - scaled_regularization_lambda() * 5.0, 2)}));

    const FloatT bias_correction = sqrt(1.0 - pow(beta2, 1)) / (1.0 - pow(beta1, 1));

    EXPECT_THAT(
        to_host(*storage->get()),
        ElementsAreArray({FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (2.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(2.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (2.5 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(2.5 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (3.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(3.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (4.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(4.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),

                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (2.0 + 10.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(2.0 + 10.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (2.5 + 11.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(2.5 + 11.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (3.0 + 12.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(3.0 + 12.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (4.0 + 13.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(4.0 + 13.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),

                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (10.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(10.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (11.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(11.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (12.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(12.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (13.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(13.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),

                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (10.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(10.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (11.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(11.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (12.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(12.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (13.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(13.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),

                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (2.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(2.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (2.5 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(2.5 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (3.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(3.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon))),
                          FPHelper<FloatT>::eq(5.0 + learning_rate() * (bias_correction * (1.0 - beta1) * (4.0 - scaled_regularization_lambda() * 5.0) / (sqrt((1.0 - beta2) * pow(4.0 - scaled_regularization_lambda() * 5.0, 2)) + epsilon)))}));
}
