#include "cuNVSM/tests_base.h"
#include "cuNVSM/cuda_utils.h"

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TEST(CudaFuncTests, truncated_sigmoid) {
    func::truncated_sigmoid<FloatT> sigmoid(0.0);

    EXPECT_THAT(sigmoid(0.0), 0.5);
    EXPECT_THAT(sigmoid(1.0), FPHelper<FloatT>::eq(0.7310585786300049));
    EXPECT_THAT(sigmoid(-1.0), FPHelper<FloatT>::eq(1.0 - 0.7310585786300049));

    EXPECT_GT(sigmoid(-50.0), 0.0);
    EXPECT_LT(sigmoid(20.0), 1.0);

    func::truncated_sigmoid<FloatT> trunc_sigmoid(1e-7);
    EXPECT_THAT(trunc_sigmoid(-100.0), FPHelper<FloatT>::eq(1e-7));
    EXPECT_THAT(trunc_sigmoid(100.0), FPHelper<FloatT>::eq(1.0 - 1e-7));
}

class CudaUtilsTest : public ::testing::TestWithParam<size_t> {
 protected:
  CudaUtilsTest() {
      // Shifts the starting address of future allocations.
      filler_.reset(new device_matrix<FloatT>(
          GetParam(), GetParam() + GetParam(), NULL));
      filler_->fillwith(NULL, GetParam());

      // Initializes the GPU memory to something different than zero.
      device_matrix<FloatT> another_filler(
          GetParam(), GetParam() + GetParam(), NULL);
      another_filler.fillwith(NULL, GetParam());
  }

  virtual void TearDown() {
      CCE(cudaDeviceSynchronize());
      CCE(cudaGetLastError());
  }

  std::unique_ptr<device_matrix<FloatT>> filler_;
};

INSTANTIATE_TEST_CASE_P(MemoryFiller,
                        CudaUtilsTest,
                        ::testing::Range<size_t>(1 /* start, inclusive */,
                                                 11 /* end, exclusive */,
                                                 1 /* step */));

TEST_P(CudaUtilsTest, Normalizer) {
    Normalizer<FloatT> normalizer(2 /* num_instances */);

    device_matrix<FloatT> input(
        5, /* num_features */
        2, /* num_instances */
        NULL /* stream */);

    to_device({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, &input);

    normalizer.forward(input, &input);

    const FloatT l2norm_first = sqrt(pow(1, 2) + pow(2, 2) + pow(3, 2) + pow(4, 2) + pow(5, 2));
    const FloatT l2norm_second = sqrt(pow(6, 2) + pow(7, 2) + pow(8, 2) + pow(9, 2) + pow(10, 2));

    EXPECT_THAT(to_host(input),
                ElementsAre(
                    1 / l2norm_first, 2 / l2norm_first, 3 / l2norm_first, 4 / l2norm_first, 5 / l2norm_first,
                    6 / l2norm_second, 7 / l2norm_second, 8 / l2norm_second, 9 / l2norm_second, 10 / l2norm_second));

    device_matrix<FloatT> grad_output(
        5, /* num_features */
        2, /* num_instances */
        NULL /* stream */);

    to_device({10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009}, &grad_output);

    std::unique_ptr<device_matrix<FloatT>> grad_input(
        normalizer.backward(grad_output));

    EXPECT_THAT(to_host(*grad_input),
                ElementsAre(980.55627996653925038,
                            612.84767497908705991,
                            245.1390699916348126,
                            -122.5695349958174063,
                            -490.27813998326962519,

                            150.11640937497926984,
                            83.398005208321819737,
                            16.679601041664362526,
                            -50.038803124993094684,
                            -116.75720729165054479));
}