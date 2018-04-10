#ifndef CUNVSM_CUDA_UTILS_H
#define CUNVSM_CUDA_UTILS_H

#include <device_matrix/device_matrix.h>

#include "base.h"

using ::cuda::FIRST_AXIS;
using ::cuda::SECOND_AXIS;
using ::cuda::DefaultStream;
using ::cuda::ScopedProfiler;
using ::cuda::Streams;
using ::cuda::Runtime;
using ::cuda::apply_elemwise;
using ::cuda::apply_columnwise;
using ::cuda::apply_except_every_Nth_column;
using ::cuda::device_matrix;
using ::cuda::fold_columns;
using ::cuda::get_scalar;
using ::cuda::make_scalar_multiplication_iterator;
using ::cuda::merge_streams;
using ::cuda::reduce_axis;

template <typename IdxType>
void generate_random_indexes(
        const IdxType max,
        const size_t num,
        RNG* const rng,
        IdxType* const output) {
    for (size_t idx = 0; idx < num; ++idx) {
        *(output + idx) = std::uniform_int_distribution<IdxType>(0, max - 1)(*rng);
    }
}

template <typename FloatT>
void init_matrix_glorot(const cudaStream_t stream,
                        device_matrix<FloatT>* const matrix,
                        RNG* const rng) {
    PROFILE_FUNCTION();

    const FloatT max = sqrt(6.0 / (matrix->getRows() + matrix->getCols()));

    FloatT* h_data = new FloatT [matrix->size()];
    for (int i = 0; i < matrix->size(); ++i) {
        h_data[i] = 2 * max * (std::generate_canonical<FloatT, 1>(*rng) - 0.5);
    }

    cudaMemcpyAsync(matrix->getData(), h_data, matrix->size() * sizeof(FloatT),
                    cudaMemcpyHostToDevice,
                    stream);

    delete [] h_data;

    CHECK_MATRIX_FINITE(*matrix);
    CHECK_MATRIX_NORM(*matrix);
}

namespace func {

using namespace ::cuda::func;

template <typename FloatT>
struct tanh {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  __host__ __device__
  FloatT operator()(const FloatT x) {
      return ::tanh(x);
  }
};

// TODO: move
// d tanh(x) / d x
template <typename FloatT>
struct tanh_to_sech2 {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  __host__ __device__
  FloatT operator()(const FloatT x) {
      return 1.0 - x * x;
  }
};

// TODO: move
// clipping in the forward pass.
template <typename FloatT>
struct clip {
  typedef FloatT argument_type;
  typedef FloatT result_type;

  clip(const FloatT min, const FloatT max, const FloatT epsilon = 1e-5)
          : actual_min_(min),
            actual_max_(max),
            min_(std::nextafter(min, min - epsilon)),
            max_(std::nextafter(max, max + epsilon)) {
      CHECK_GT(max, min);
  }

  __host__ __device__
  FloatT operator()(const FloatT x) {
      return min(max(x, min_), max_);
  }

 private:
  // For edge-case debugging.
  const FloatT actual_min_;
  const FloatT actual_max_;

  const FloatT min_;
  const FloatT max_;
};

// TODO: move
// derivative of func::clip, working on its output.
//
// the implementation is a slight hack, as we do not have the input
// to the clipping operation. Therefore, we just assume that if a value
// is extreme (i.e. either min_ or max_), it had been clipped.
template <typename FloatT>
struct clip_to_clip_deriv {
  typedef FloatT argument_type;
  typedef FloatT result_type;

  clip_to_clip_deriv(const FloatT min, const FloatT max,
                     const FloatT epsilon = 1e-5)
          : actual_min_(min),
            actual_max_(max),
            min_(std::nextafter(min, min - epsilon)),
            max_(std::nextafter(max, max + epsilon)) {
      CHECK_GT(max, min);
  }

  __host__ __device__
  FloatT operator()(const FloatT x) {
      const FloatT dx = (x > min_ && x < max_) ? 1.0 : 0.0;

      return dx;
  }

 private:
  // For edge-case debugging.
  const FloatT actual_min_;
  const FloatT actual_max_;

  const FloatT min_;
  const FloatT max_;
};

template <typename FloatT>
struct log {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  __host__ __device__
  FloatT operator()(const FloatT x) {
      return ::log(x);
  }
};

// TODO: move
template <typename FloatT>
struct prob_complement {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  __host__ __device__
  FloatT operator() (const FloatT& x) {
      return 1.0 - x;
  }
};

// TODO: move
template <typename FloatT>
struct mutiply_with_complement_prob {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  __host__ __device__
  FloatT operator()(const FloatT x, const FloatT prob) {
      return x * (1.0 - prob);
  }
};

template <typename FloatT>
struct sqrt {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  __host__ __device__
  FloatT operator()(const FloatT& x) {
      return ::sqrt(x);
  }
};

// TODO: move
template <typename FloatT>
struct truncated_sigmoid {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  explicit truncated_sigmoid(const FloatT epsilon)
          : epsilon_(epsilon) {
      CHECK_GE(epsilon_, 0.0);
  }

  __host__ __device__
  FloatT operator() (const FloatT& x) {
      // Numerically stable sigmoid;
      // from http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
      const FloatT prob = (x >= 0)
          ? 1.0 / (1.0 + exp(-x))
          : exp(x) / (1.0 + exp(x));

      return min(max(prob, epsilon_), 1.0 - epsilon_);
  }

 private:
  const FloatT epsilon_;
};

// TODO: move
template <typename FloatT>
struct sigmoid_to_log_sigmoid_deriv {
  typedef FloatT argument_type;
  typedef FloatT result_type;
  explicit sigmoid_to_log_sigmoid_deriv(const FloatT epsilon)
          : epsilon_(epsilon) {
      CHECK_GE(epsilon_, 0.0);
  }

  // d log(sigmoid(x)) / dx = (1 / sigmoid(x)) * sigmoid(x) * (1.0 - sigmoid(x))
  //                        = (1.0 - sigmoid(x))
  __host__ __device__
  FloatT operator() (const FloatT& x) {
      return (x >= (1.0 - epsilon_) || x <= epsilon_) ? 0.0 : 1.0 - x;
  }

 private:
  const FloatT epsilon_;
};

}  // namespace func

template <typename FloatT>
FloatT l2_norm(const device_matrix<FloatT>& dmat) {
    PROFILE_FUNCTION();

    return sqrt(thrust::transform_reduce(
        begin(dmat),
        end(dmat),
        func::square<FloatT>(),
        0.0,
        thrust::plus<FloatT>()));
}

// TODO: move
// Not optimized; but that's okay as this is not used during training.
template <typename FloatT>
void inplace_l2_normalize_columns(device_matrix<FloatT>* const dmat) {
    // TODO(cvangysel): get rid of extra copy here by building in an element-wise
    // operation into the reduce_axis kernel.
    std::unique_ptr<device_matrix<FloatT>> dmat_copy(
        dmat->copy(dmat->getStream()));

    apply_elemwise<func::square<FloatT>>(
        thrust::cuda::par.on(dmat_copy->getStream()),
        dmat_copy.get());

    device_matrix<FloatT> sum_of_squares(1, /* num_rows */
                                         dmat_copy->getCols(),
                                         dmat_copy->getStream());

    reduce_axis(sum_of_squares.getStream(),
                FIRST_AXIS,
                *dmat_copy,
                &sum_of_squares);

    apply_elemwise<func::sqrt<FloatT>>(
        thrust::cuda::par.on(sum_of_squares.getStream()),
        &sum_of_squares);

    apply_columnwise<thrust::divides<FloatT>>(
        thrust::cuda::par.on(sum_of_squares.getStream()),
        sum_of_squares,
        dmat);
}

template <typename FloatT>
void increment_scalar(/* __host__ */const FloatT value,
                      /* __device__ */ FloatT* const scalar_ptr) {
    const FloatT old_value = get_scalar(scalar_ptr);
    const FloatT new_value = old_value + value;

    DLOG_EVERY_N(WARNING, 1000) << "Call to increment_scalar can cripple performance.";

    cudaMemcpy(scalar_ptr,
               &new_value,
               sizeof(FloatT),
               cudaMemcpyHostToDevice);

    DCHECK_EQ(get_scalar(scalar_ptr), new_value);
}

// TODO: move
template <typename FloatT>
class Normalizer {
 public:
  Normalizer(const size_t num_instances);

  void forward(const device_matrix<FloatT>& input,
               device_matrix<FloatT>* const output);

  void backward(const device_matrix<FloatT>& grad_output,
                device_matrix<FloatT>* const grad_input);

  device_matrix<FloatT>* backward(const device_matrix<FloatT>& grad_output);

 private:
  const size_t num_instances_;

  std::unique_ptr<device_matrix<FloatT>> input_cache_;
  std::unique_ptr<device_matrix<FloatT>> norms_;

  DISALLOW_COPY_AND_ASSIGN(Normalizer);
};

#endif /* CUNVSM_CUDA_UTILS_H */