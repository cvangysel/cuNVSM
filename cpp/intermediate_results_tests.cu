#include "cuNVSM/tests_base.h"
#include "cuNVSM/intermediate_results.h"

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TEST(IntermediateResultsTest, MergeGradientsFn) {
    MergeGradientsFn<FloatT> merge_gradients_fn;

    std::unique_ptr<Gradients<FloatT>> first_gradients(new SingleGradients<FloatT>(nullptr));

    std::unique_ptr<Gradients<FloatT>> second_gradients(new SingleGradients<FloatT>(nullptr));

    std::unique_ptr<Gradients<FloatT>> merged_gradients(
        merge_gradients_fn({
            std::make_pair<Gradients<FloatT>*, FloatT>(first_gradients.release(), 0.5),
            std::make_pair<Gradients<FloatT>*, FloatT>(second_gradients.release(), 0.5),
        }));
}