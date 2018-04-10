#ifndef CUNVSM_CUDNN_H
#define CUNVSM_CUDNN_H

#include <cudnn.h>

#include "cuNVSM/cuda_utils.h"

void checkCuDNNErrors(const cudnnStatus_t status,
                      const char* const filename, const int line);

#define CCDNNE(val) checkCuDNNErrors(val, __FILE__, __LINE__)

template <typename FloatT>
class CuDNNDataType {
 public:
  CuDNNDataType() {}
};

template <>
class CuDNNDataType<float32> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};

template <>
class CuDNNDataType<float64> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
};

template <typename FloatT>
class CuDNNRuntime {
 public:
  static CuDNNRuntime* getInstance() {
      return INSTANCE_;
  }

  inline cudnnHandle_t& handle() {
      return handle_;
  }

  inline cudnnTensorDescriptor_t* build_descriptor(const device_matrix<FloatT>& matrix) {
      cudnnTensorDescriptor_t* tensor_handle = new cudnnTensorDescriptor_t;
      CCDNNE(cudnnCreateTensorDescriptor(tensor_handle));

      std::vector<int> size(4);
      size[0] = matrix.getCols();
      size[1] = matrix.getRows();
      size[2] = 1;
      size[3] = 1;

      std::vector<int> stride(4);
      stride[0] = matrix.getRows();
      stride[1] = 1;
      stride[2] = 1;
      stride[3] = 1;

      CCDNNE(cudnnSetTensorNdDescriptor(
             *tensor_handle,
             CuDNNDataType<FloatT>::type,
             4, /* nbDims */
             size.data(),
             stride.data()));

      return tensor_handle;
  }
 private:
  CuDNNRuntime() {
      CCDNNE(cudnnCreate(&handle_));
  }

  ~CuDNNRuntime() {
      CCDNNE(cudnnDestroy(handle_));
  }

  static CuDNNRuntime* INSTANCE_;

  cudnnHandle_t handle_;

  DISALLOW_COPY_AND_ASSIGN(CuDNNRuntime);
};

template <typename FloatT>
class BatchNormalization {
 public:
  explicit BatchNormalization(const size_t num_features,
                              const FloatT momentum = 0.1,
                              const FloatT epsilon = 1e-4,
                              const bool cache_input = false);

  virtual ~BatchNormalization();

  void forward(const device_matrix<FloatT>& input,
               const device_matrix<FloatT>& bias,
               device_matrix<FloatT>* const output);

  void backward(const device_matrix<FloatT>& grad_output,
                const device_matrix<FloatT>& bias,
                device_matrix<FloatT>* const grad_input,
                device_matrix<FloatT>* const grad_bias);

  void backward(const device_matrix<FloatT>& grad_output,
                const device_matrix<FloatT>& input,
                const device_matrix<FloatT>& bias,
                device_matrix<FloatT>* const grad_input,
                device_matrix<FloatT>* const grad_bias);

 private:
  const size_t num_features_;

  const FloatT momentum_;
  const FloatT epsilon_;

  const bool cache_input_;

  std::unique_ptr<device_matrix<FloatT>> input_cache_;

  std::unique_ptr<device_matrix<FloatT>> gamma_;
  std::unique_ptr<device_matrix<FloatT>> grad_gamma_;

  std::unique_ptr<device_matrix<FloatT>> mean_cache_;
  std::unique_ptr<device_matrix<FloatT>> inv_variance_cache_;

  std::unique_ptr<cudnnTensorDescriptor_t> io_desc_;
  std::unique_ptr<cudnnTensorDescriptor_t> stats_desc_;

  DISALLOW_COPY_AND_ASSIGN(BatchNormalization);
};

#endif /* CUNVSM_CUDNN_H */