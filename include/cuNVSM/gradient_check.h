#ifndef CUNVSM_GRADIENT_CHECK_H
#define CUNVSM_GRADIENT_CHECK_H

#include "cuNVSM/model.h"

template <typename ModelT>
class GradientCheckFn {
 public:
  typedef typename ModelT::FloatT FloatT;

  bool operator()(ModelT* const model,
                  const typename ModelT::Batch& batch,
                  const typename ModelT::ForwardResult& result,
                  const typename ModelT::Gradients& gradients,
                  const FloatT epsilon,
                  const FloatT relative_error_threshold,
                  const std::stringstream& rng_state,
                  RNG* const rng);
};

#endif /* CUNVSM_GRADIENT_CHECK_H */