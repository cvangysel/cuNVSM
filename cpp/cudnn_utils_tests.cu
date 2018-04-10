#include "cuNVSM/tests_base.h"
#include "cuNVSM/cudnn_utils.h"

using ::testing::ElementsAreArray;
using ::testing::ElementsAre;

class cuDNNTests : public ::testing::Test {
  virtual void SetUp() {
      ::testing::Test::SetUp();
      VLOG(3) << "Runtime: " << Runtime<FloatT>::getInstance();
  }

  virtual void TearDown() {
      CCE(cudaDeviceSynchronize());
      CCE(cudaGetLastError());
  }
};

TEST_F(cuDNNTests, BatchNormalization_large) {
    BatchNormalization<FloatT> bn(10 /* num_features */);

    device_matrix<FloatT> input(
        10, /* num_features*/
        100, /* num_instances */
        NULL /* stream */);
    input.fillwith(input.getStream(), 1.0);

    device_matrix<FloatT> bias(10, 1, NULL /* stream */);
    bias.fillwith(bias.getStream(), 0.0);

    bn.forward(input, bias, &input);

    EXPECT_THAT(
        to_host(input),
        ::testing::Each(FPHelper<FloatT>::eq(0.0)));
}

TEST_F(cuDNNTests, BatchNormalization_very_large) {
    const size_t num_features = 256;
    const size_t num_instances = 100000;

    BatchNormalization<FloatT> first_bn(num_features);
    BatchNormalization<FloatT> second_bn(num_features);

    device_matrix<FloatT> input(
        num_features,
        num_instances,
        NULL /* stream */);

    {
        std::vector<FloatT> data;
        for (size_t i = 0; i < num_features * num_instances; ++i) {
            data.push_back(static_cast<FloatT>(i));
        }

        input.fillwith(input.getStream(), data);
    }

    std::unique_ptr<device_matrix<FloatT>> second_input(
        input.copy(input.getStream()));

    device_matrix<FloatT> bias(num_features, 1, NULL /* stream */);
    bias.fillwith(bias.getStream(), 0.0);

    device_matrix<FloatT> output(
        num_features,
        num_instances,
        NULL /* stream */);

    {
        first_bn.forward(input, bias, &input);
        second_bn.forward(*second_input, bias, &output);
    }

    EXPECT_EQ(to_host(input), to_host(output));

    device_matrix<FloatT> first_grad_input(
        num_features,
        num_instances,
        NULL /* stream */);
    first_grad_input.fillwith(first_grad_input.getStream(), 1.0);
    device_matrix<FloatT> grad_output(
        num_features,
        num_instances,
        NULL /* stream */);
    grad_output.fillwith(grad_output.getStream(), 1.0);

    device_matrix<FloatT> second_grad_input(
        num_features,
        num_instances,
        NULL /* stream */);

    device_matrix<FloatT> first_grad_bias(num_features, 1, NULL /* stream */);
    device_matrix<FloatT> second_grad_bias(num_features, 1, NULL /* stream */);

    {
        first_bn.backward(first_grad_input,
                          *second_input,
                          bias,
                          &first_grad_input,
                          &first_grad_bias);
        second_bn.backward(grad_output,
                           *second_input,
                           bias,
                           &second_grad_input,
                           &second_grad_bias);
    }

    print_matrix(second_grad_input);

    EXPECT_EQ(to_host(first_grad_input), to_host(second_grad_input));
    EXPECT_EQ(to_host(first_grad_bias), to_host(second_grad_bias));
}

TEST_F(cuDNNTests, BatchNormalization_forward_backward) {
    const FloatT epsilon = 1e-5;

    BatchNormalization<FloatT> bn(3, /* num_features */
                                  0.1, /* momentum */
                                  epsilon);

    // Construct parameter(s).
    device_matrix<FloatT> bias(3, 1, NULL /* stream */);
    bias.fillwith(bias.getStream(), 0.0);

    // Construct input.
    device_matrix<FloatT> input(
        3, /* num_features*/
        2, /* num_instances */
        NULL /* stream */);
    input.fillwith(input.getStream(),
                   {1.0, 2.0, 3.0,
                    5.0, 10.0, 20.0});

    device_matrix<FloatT> output(
        3, /* num_features*/
        2, /* num_instances */
        NULL /* stream */);

    // Forward pass.
    bn.forward(input, bias, &output);

    EXPECT_THAT(
        to_host(output),
        ElementsAreArray({
            (1.0 - 3.0) / sqrt(4.0 + epsilon), (2.0 - 6.0) / sqrt(16.0 + epsilon), (3.0 - 11.5) / sqrt(72.25 + epsilon),
            (5.0 - 3.0) / sqrt(4.0 + epsilon), (10.0 - 6.0) / sqrt(16.0 + epsilon), (20.0 - 11.5) / sqrt(72.25 + epsilon),
        }));

    device_matrix<FloatT> grad(
        3, /* num_features*/
        2, /* num_instances */
        NULL /* stream */);
    grad.fillwith(grad.getStream(),
                  {0.25, -0.1, 0.3,
                   1.0, 0.005, -0.5});

    device_matrix<FloatT> grad_bias(3, 1, NULL /* stream */);

    bn.backward(grad, input, bias,
                &grad, &grad_bias);

    EXPECT_THAT(
        to_host(grad_bias),
        ElementsAre(FPHelper<FloatT>::eq(1.25),
                    FPHelper<FloatT>::eq(-0.09500000000000000111),
                    FPHelper<FloatT>::eq(-0.2000000000000000111)));

    EXPECT_THAT(
        to_host(grad),
        ElementsAre(FPHelper<FloatT>::eq(-4.687482422216504574e-07),
                    FPHelper<FloatT>::eq(-8.2031173104235577398e-09),
                    FPHelper<FloatT>::eq(6.5133306248466027455e-09),
                    FPHelper<FloatT>::eq(4.687482422216504574e-07),
                    FPHelper<FloatT>::eq(8.2031173086888342638e-09),
                    FPHelper<FloatT>::eq(-6.5133306248466027455e-09)));
}