#include "cuNVSM/cudnn_utils.h"

static const FLOATING_POINT_TYPE ZERO_ = 0.0;
static const FLOATING_POINT_TYPE ONE_ = 1.0;

void checkCuDNNErrors(const cudnnStatus_t status,
                      const char* const filename, const int line) {
    if (status == CUDNN_STATUS_SUCCESS) {
        return;
    }

    std::string error;

    switch (status) {
      case CUDNN_STATUS_NOT_INITIALIZED:
          error = "CUDNN_STATUS_NOT_INITIALIZED";
          break;
      case CUDNN_STATUS_ALLOC_FAILED:
          error = "CUDNN_STATUS_ALLOC_FAILED";
          break;
      case CUDNN_STATUS_BAD_PARAM:
          error = "CUDNN_STATUS_BAD_PARAM";
          break;
      case CUDNN_STATUS_ARCH_MISMATCH:
          error = "CUDNN_STATUS_ARCH_MISMATCH";
          break;
      case CUDNN_STATUS_MAPPING_ERROR:
          error = "CUDNN_STATUS_MAPPING_ERROR";
          break;
      case CUDNN_STATUS_EXECUTION_FAILED:
          error = "CUDNN_STATUS_EXECUTION_FAILED";
          break;
      case CUDNN_STATUS_INTERNAL_ERROR:
          error = "CUDNN_STATUS_INTERNAL_ERROR";
          break;
      case CUDNN_STATUS_NOT_SUPPORTED:
          error = "CUDNN_STATUS_NOT_SUPPORTED";
          break;
      case CUDNN_STATUS_LICENSE_ERROR:
          error = "CUDNN_STATUS_LICENSE_ERROR";
          break;
      default:
          error = "UNKNOWN";
    };

    LOG(FATAL) << "CuDNN error " << error << " in " << filename << " at line " << line << ".";
}

template <typename FloatT>
BatchNormalization<FloatT>::BatchNormalization(const size_t num_features,
                                               const FloatT momentum,
                                               const FloatT epsilon,
                                               const bool cache_input)
        : num_features_(num_features),
          momentum_(momentum), epsilon_(epsilon),
          cache_input_(cache_input),
          input_cache_(nullptr),
          gamma_(new device_matrix<FloatT>(
              1, num_features_, NULL /* stream */)),
          grad_gamma_(new device_matrix<FloatT>(
              1, num_features_, NULL /* stream */)),
          mean_cache_(new device_matrix<FloatT>(
              1, num_features_, NULL /* stream */)),
          inv_variance_cache_(new device_matrix<FloatT>(
              1, num_features_, NULL /* stream */)) {
    CHECK_GE(epsilon_, CUDNN_BN_MIN_EPSILON);

    gamma_->fillwith(gamma_->getStream(), 1.0);
}

template <typename FloatT>
BatchNormalization<FloatT>::~BatchNormalization() {
    if (io_desc_ != nullptr) {
        cudnnDestroyTensorDescriptor(*io_desc_);
    }

    if (stats_desc_ != nullptr) {
        cudnnDestroyTensorDescriptor(*stats_desc_);
    }
}

template <typename FloatT>
void BatchNormalization<FloatT>::forward(
        const device_matrix<FloatT>& input,
        const device_matrix<FloatT>& bias,
        device_matrix<FloatT>* const output) {
    CHECK_EQ(input.getRows(), num_features_);
    CHECK_DIMENSIONS(bias, num_features_, 1);

    if (io_desc_ == nullptr) {
        io_desc_.reset(
            CuDNNRuntime<FloatT>::getInstance()->build_descriptor(input));
    }
    if (stats_desc_ == nullptr) {
        stats_desc_.reset(
            CuDNNRuntime<FloatT>::getInstance()->build_descriptor(bias));
    }

    if (cache_input_) {
        if (input_cache_ == nullptr || !input_cache_->hasSameShape(input)) {
            input_cache_.reset(input.copy(input.getStream()));
        } else {
            input_cache_->copyFrom(input_cache_->getStream(), input);
        }
    }

    CCDNNE(cudnnBatchNormalizationForwardTraining(
        CuDNNRuntime<FloatT>::getInstance()->handle(),
        CUDNN_BATCHNORM_PER_ACTIVATION,
        &ONE_, /* alpha; scaling for result */
        &ZERO_, /* beta; scaling for existing value in output */
        *io_desc_,
        raw_begin(input), /* x */
        *io_desc_,
        raw_begin(*output), /* y */
        *stats_desc_,
        raw_begin(*gamma_), /* bnScale */
        raw_begin(bias), /* bnBias */
        1.0, /* exponential_average_factor */
        nullptr, /* result_running_mean */
        nullptr, /* result_running_inv_variance */
        epsilon_,
        raw_begin(*mean_cache_),
        raw_begin(*inv_variance_cache_)));

    CCE(cudaGetLastError());

    CHECK_MATRIX(*output);
}

template <typename FloatT>
void BatchNormalization<FloatT>::backward(
        const device_matrix<FloatT>& grad_output,
        const device_matrix<FloatT>& bias,
        device_matrix<FloatT>* const grad_input,
        device_matrix<FloatT>* const grad_bias) {
    CHECK(cache_input_);
    CHECK(input_cache_ != nullptr);

    backward(grad_output, *input_cache_, bias, grad_input, grad_bias);
}

template <typename FloatT>
void BatchNormalization<FloatT>::backward(
        const device_matrix<FloatT>& grad_output,
        const device_matrix<FloatT>& input,
        const device_matrix<FloatT>& bias,
        device_matrix<FloatT>* const grad_input,
        device_matrix<FloatT>* const grad_bias) {
    CHECK_EQ(grad_output.getRows(), num_features_);
    CHECK_EQ(input.getRows(), num_features_);
    CHECK_EQ(grad_output.getCols(), input.getCols());
    CHECK_DIMENSIONS(bias, num_features_, 1);

    CHECK(io_desc_ != nullptr);
    CHECK(stats_desc_ != nullptr);

    CCDNNE(cudnnBatchNormalizationBackward(
        CuDNNRuntime<FloatT>::getInstance()->handle(),
        CUDNN_BATCHNORM_PER_ACTIVATION,
        &ONE_, /* alphaDataDiff */
        &ZERO_, /* betaDataDiff */
        &ONE_, /* alphaParamDiff */
        &ZERO_, /* betaParamDiff */
        *io_desc_,
        raw_begin(input),
        *io_desc_,
        raw_begin(grad_output),
        *io_desc_,
        raw_begin(*grad_input),
        *stats_desc_,
        raw_begin(*gamma_),
        raw_begin(*grad_gamma_),
        raw_begin(*grad_bias),
        epsilon_,
        raw_begin(*mean_cache_),
        raw_begin(*inv_variance_cache_)));

    CCE(cudaGetLastError());

    CHECK_MATRIX(*grad_input);
    CHECK_MATRIX(*grad_bias);
}

// Explicit instantiations.
template <>
CuDNNRuntime<FLOATING_POINT_TYPE>* CuDNNRuntime<FLOATING_POINT_TYPE>::INSTANCE_ = new CuDNNRuntime;
template class BatchNormalization<FLOATING_POINT_TYPE>;
