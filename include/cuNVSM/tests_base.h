#ifndef CUNVSM_TESTS_BASE_H
#define CUNVSM_TESTS_BASE_H

#include <algorithm>
#include <cmath>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <google/protobuf/text_format.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "cuNVSM/base.h"
#include "cuNVSM/data.h"

typedef FLOATING_POINT_TYPE FloatT;
typedef int32 IdxType;

template <typename FloatT>
class FPHelper {};
template <>
class FPHelper<float32> {
 public:
  static const decltype(&::testing::FloatEq) eq;
};
template <>
class FPHelper<float64> {
 public:
  static const decltype(&::testing::DoubleEq) eq;
};

const decltype(&::testing::FloatEq) FPHelper<float32>::eq = &::testing::FloatEq;
const decltype(&::testing::DoubleEq) FPHelper<float64>::eq = &::testing::DoubleEq;

template <typename T>
T ParseProto(const std::string& msg) {
    T proto;
    CHECK(google::protobuf::TextFormat::ParseFromString(msg, &proto));

    return proto;
}

// CUDA-specific helpers.
#ifdef NVCC
#include "cuda_utils.h"

template <typename FloatT>
std::vector<FloatT> to_host(const device_matrix<FloatT>& device_mat) {
    FloatT* data = get_array(device_mat.getStream(), device_mat);
    std::vector<FloatT> v(data, data + device_mat.size());
    delete [] data;

    return v;
}

template <typename FloatT>
std::vector<bool> to_host_isfinite(const device_matrix<FloatT>& device_mat) {
    std::vector<FloatT> data = to_host(device_mat);

    std::vector<bool> data_isfinite;

    for (const FloatT x : data) {
        data_isfinite.push_back(std::isfinite(x));
    }

    return data_isfinite;
}

template <typename FloatT>
void to_device(const std::vector<FloatT>& data, device_matrix<FloatT>* const matrix) {
    matrix->fillwith(matrix->getStream(), data);
}
#endif

class DummyTextEntityDataSource : public TextEntity::DataSource {
 public:
  DummyTextEntityDataSource()
      : TextEntity::DataSource(11 /* vocabulary_size */,
                               11 /* corpus_size */),
        num_batches_emitted_(0) {
      reset();
  }

  virtual void reset() override {
      num_batches_emitted_ = 0;
  }

  virtual void next(TextEntity::Batch* const batch) override {
      TextEntity::DataSource::next(batch);

      for (size_t idx = 0; idx < batch->maximum_size(); ++idx) {
          std::vector<WordIdxType> features(batch->window_size());
          std::vector<FloatT> feature_weights(batch->window_size());

          for (size_t i = 0; i < features.size(); ++i) {
              features[i] = get_feature_value();
              feature_weights[i] = get_feature_weight();
          }

          push_instance(features,
                        feature_weights,
                        get_label(), /* label */
                        get_weight() /* weight */,
                        batch);
      }

      ++num_batches_emitted_;
  }

  virtual bool has_next() const override {
      return TextEntity::DataSource::has_next() || num_batches_emitted_ < 2;
  }

 protected:
  virtual WordIdxType get_feature_value() = 0;
  virtual FloatT get_feature_weight() = 0;
  virtual ObjectIdxType get_label() = 0;
  virtual FloatT get_weight() = 0;

 private:
  size_t num_batches_emitted_;

  DISALLOW_COPY_AND_ASSIGN(DummyTextEntityDataSource);
};

class ConstantSource : public DummyTextEntityDataSource {
 public:
  ConstantSource(const WordIdxType feature_value = 10,
                 const ObjectIdxType label = 10)
      : DummyTextEntityDataSource(), feature_value_(feature_value), label_(label) {}

 protected:
  virtual WordIdxType get_feature_value() override {
      return feature_value_;
  }

  virtual FloatT get_feature_weight() override {
      return 1.0;
  }

  virtual ObjectIdxType get_label() override {
      return label_;
  }

  virtual FloatT get_weight() override {
      return 1.0;
  }

 private:
  const WordIdxType feature_value_;
  const ObjectIdxType label_;

  DISALLOW_COPY_AND_ASSIGN(ConstantSource);
};

class RandomSource : public DummyTextEntityDataSource {
 public:
  explicit RandomSource(RNG* const rng)
      : DummyTextEntityDataSource(),
        rng_(rng),
        distribution_(0, 10), weight_distribution_(0.0, 2.0) {}

 protected:
  virtual WordIdxType get_feature_value() override {
      return distribution_(*rng_);
  }

  virtual FloatT get_feature_weight() override {
      return weight_distribution_(*rng_);
  }

  virtual ObjectIdxType get_label() override {
      return distribution_(*rng_);
  }

  virtual FloatT get_weight() override {
      return weight_distribution_(*rng_);
  }

 private:
  RNG* rng_;

  std::uniform_int_distribution<int32> distribution_;
  std::uniform_real_distribution<FloatT> weight_distribution_;

  DISALLOW_COPY_AND_ASSIGN(RandomSource);
};

#endif /* CUNVSM_TESTS_BASE_H */