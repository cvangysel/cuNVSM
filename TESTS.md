Unit and integration tests
==========================

cuNVSM comes with a test harness to verify its implementation. Here we give a brief overview of the tests:

   * [cuda\_utils\_tests](cpp/cuda_utils_tests.cu) verifies the implementation of custom CUDA kernels and transformation functions (e.g., `truncated_sigmoid`). Note that most CUDA-related functionality is tested by the [device\_matrix](https://github.com/cvangysel/device_matrix) library.
   * [cudnn\_utils\_tests](cpp/cudnn_utils_tests.cu) tests the integration of cuDNN within cuNVSM and, in particular, batch normalization.
   * [data\_tests](cpp/data_tests.cpp) checks whether all the data loading works properly. For example, it contains tests that verify the integration with Indri.
   * [gradient\_checking\_tests](cpp/gradient_checking_tests.cu) provides automated tests, using numerical differentiation, that verify whether the forward and backward passes are correctly implemented. If a change to the loss functions is made, the consistency can be verified using this module.
   * [intermediate\_results\_tests](cpp/intermediate_results_tests.cu) provides shallow tests for verifying the correct workings of internal cuNVSM data structures.
   * [model\_tests](cpp/model_tests.cu) is a mixed bag of tests that verify the interface to the `Model` class.
   * [updates\_tests](cpp/updates_tests.cu) verifies the implementation of parameter update methods (e.g., [Adam](https://arxiv.org/abs/1412.6980)).
   * [utils\_tests](cpp/utils\_tests.cpp) contains a few short tests that verify auxiliary functions.