#ifndef CUNVSM_BASE_H
#define CUNVSM_BASE_H

#include <algorithm>
#include <deque>
#include <queue>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <random>
#include <utility>

#include <glog/logging.h>

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "c++14.h"

enum ParamIdentifier {
    WORD_REPRS, TRANSFORM, ENTITY_REPRS
};

typedef unsigned short uint16;

typedef long int32;
typedef unsigned long uint32;
typedef long long int64;
typedef unsigned long long uint64;

typedef float float32;
typedef double float64;

typedef std::minstd_rand0 RNG;

// Makes class instantiations non-copyable.
//
// From StackOverflow.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  void operator=(const TypeName&)

// Adds a constructor to a class that takes a single tuple and
// forwards the arguments to the class constructor.
//
// From StackOverflow.
//
// TODO(cvangysel): currently unused as it was easier to add the
//                  constructors manually.
#define CONSTRUCT_FROM_TUPLE(TypeName)                    \
  template <class... Ts>                                  \
  TypeName(std::tuple<Ts...> const& tup)                  \
      : TypeName(tup, index_sequence_for<Ts...>{})   \
  {}                                                      \
                                                          \
  template <class Tuple, size_t... Is>                    \
  TypeName(Tuple const& tup, index_sequence<Is...> ) \
      : TypeName(std::get<Is>(tup)...)                    \
  {}                                                      \
  int _____________SYNTACTIC_SUGAR[0] /* To avoid warnings about additional semi-colon */

#define __LSE_DEBUG 0

#ifndef FLOATING_POINT_TYPE
#pragma error("FLOATING_POINT_TYPE macro should be set.")
#endif

#define STR(x) #x
#define SHOW_DEFINE(x) #x << "=" << STR(x)

template <typename T>
class TopN {
 public:
  explicit TopN(const size_t n) : n_(n) {
      CHECK_GT(n_, 0);
  }

  void push(const T& item) {
      if (pq_.size() >= n_ && pq_.top() < item) {
          pq_.pop();
      }

      if (pq_.size() < n_) {
          pq_.push(item);
      }
  }

  void get_data(std::vector<T>* const data) {
      CHECK(data->empty());

      while (!pq_.empty()) {
          data->push_back(pq_.top());
          pq_.pop();
      }
  }

 protected:
  const size_t n_;
  std::priority_queue<T, std::vector<T>, std::greater<T>> pq_;
};

inline std::vector<std::string> split(const std::string& str) {
    std::istringstream iss(str);

    return std::vector<std::string>(
        std::istream_iterator<std::string>{iss},
        std::istream_iterator<std::string>{});
}

inline std::ostream& operator<<(std::ostream& os, const google::protobuf::Message& message) {
    std::string str;
    CHECK(google::protobuf::TextFormat::PrintToString(message, &str));

    os << str;

    return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << std::setprecision(20) << "[";
    copy(v.begin(), v.end(), std::ostream_iterator<T>(os, ", "));
    os << "]";

    return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& v) {
    os << "[";
    for (const auto& e : v) {
        os << e << ", ";
    }
    os << "]";

    return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::deque<T>& v) {
    os << "[";
    copy(v.begin(), v.end(), std::ostream_iterator<T>(os, ", "));
    os << "]";

    return os;
}

template <typename T>
void flatten(const std::vector<std::vector<T> >& iterable,
             std::vector<T>* const flattened) {
    CHECK(flattened->empty());

    for (const auto& instance : iterable) {
        std::copy(instance.begin(), instance.end(),
                  std::back_inserter(*flattened));
    }
}

template <typename FloatT>
inline std::vector<FloatT> range(size_t start, size_t end, size_t repeat = 1) {
    std::vector<FloatT> v;

    for (size_t i = start; i < end; ++i) {
        for (size_t j = 0; j < repeat; ++j) v.push_back(i);
    }

    return v;
}

template <typename Iterable>
inline bool contains_key(const Iterable& iterable,
                         const typename Iterable::key_type& key) {
    return iterable.find(key) != iterable.end();
}

template <typename ContainerT>
inline bool insert_or_die(const typename ContainerT::value_type& value,
                          ContainerT* const container) {
    CHECK(!contains_key(*container, value));
    container->insert(value);

    return true;
}

template <typename ValueT>
inline bool insert_or_die(const ValueT& value,
                          std::vector<ValueT>* const container) {
    container->push_back(value);

    return true;
}

template <typename MapT>
inline bool insert_or_die(const typename MapT::key_type& key,
                          const typename MapT::mapped_type& value,
                          MapT* const container) {
    CHECK(!contains_key(*container, key));
    container->insert(std::make_pair(key, value));

    return true;
}

template <typename MapT>
inline bool insert_or_update(const typename MapT::key_type& key,
                             const typename MapT::mapped_type& value,
                             MapT* const container) {
    if (contains_key(*container, key)) {
        container->find(key)->second = value;
    } else {
        insert_or_die(key, value, container);
    }

    return true;
}

template <typename MapT>
typename MapT::mapped_type find_with_default(
        const MapT& container,
        const typename MapT::key_type& key,
        const typename MapT::mapped_type& default_value) {
    if (contains_key(container, key)) {
        return container.at(key);
    } else {
        return default_value;
    }
}

inline std::string seconds_to_humanreadable_time(float64 seconds) {
    const int32 hours = floor(seconds / 3600.0);
    seconds -= hours * 3600.0;

    const int32 minutes = floor(seconds / 60.0);
    seconds -= minutes * 60.0;

    std::stringstream stream;
    stream << hours << " hours, ";
    stream << minutes << " minutes and ";
    stream << static_cast<int32>(floor(seconds)) << " seconds";

    return stream.str();
}

inline bool is_number(const std::string& s) {
    return !s.empty() &&
           std::find_if(s.begin(), s.end(), (int(*) (int)) std::isdigit) != s.end();
}

#if 1 // Not in C++11 // make_index_sequence
#include <cstdint>
namespace std {
template <std::size_t...> struct index_sequence {};

template <std::size_t N, std::size_t... Is>
struct make_index_sequence : make_index_sequence<N - 1, N - 1, Is...> {};

template <std::size_t... Is>
struct make_index_sequence<0u, Is...> : index_sequence<Is...> { using type = index_sequence<Is...>; };
}  // namespace std
#endif // make_index_sequence

#endif /* CUNVSM_BASE_H */