#ifndef CUNVSM_CPP14_H
#define CUNVSM_CPP14_H

// From https://github.com/taocpp/sequences/blob/master/include/tao/seq/integer_sequence.hpp.
template< typename T, T... Ns >
struct integer_sequence {
    using value_type = T;

    static constexpr std::size_t size() noexcept {
        return sizeof...( Ns );
    }
};
template< std::size_t... Ns >
using index_sequence = integer_sequence< std::size_t, Ns... >;

// From https://github.com/taocpp/sequences/blob/master/include/tao/seq/make_integer_sequence.hpp
namespace impl {
  template< typename, std::size_t, bool >
  struct double_up;

  template< typename T, T... Ns, std::size_t N >
  struct double_up< integer_sequence< T, Ns... >, N, false >
  {
    using type = integer_sequence< T, Ns..., ( N + Ns )... >;
  };

  template< typename T, T... Ns, std::size_t N >
  struct double_up< integer_sequence< T, Ns... >, N, true >
  {
    using type = integer_sequence< T, Ns..., ( N + Ns )..., 2 * N >;
  };

  template< typename T, T N, typename = void >
  struct generate_sequence;

  template< typename T, T N >
  using generate_sequence_t = typename generate_sequence< T, N >::type;

  template< typename T, T N, typename >
  struct generate_sequence
    : double_up< generate_sequence_t< T, N / 2 >, N / 2, N % 2 == 1 >
  {};

  template< typename T, T N >
  struct generate_sequence< T, N, typename std::enable_if< ( N == 0 ) >::type >
  {
    using type = integer_sequence< T >;
  };

  template< typename T, T N >
  struct generate_sequence< T, N, typename std::enable_if< ( N == 1 ) >::type >
  {
    using type = integer_sequence< T, 0 >;
  };
}

template< typename T, T N >
using make_integer_sequence = impl::generate_sequence_t< T, N >;

template< std::size_t N >
using make_index_sequence = make_integer_sequence< std::size_t, N >;

template< typename... Ts >
using index_sequence_for = make_index_sequence< sizeof...( Ts ) >;

template<int...> struct index_tuple{}; 

template <size_t Idx>
struct tuple_index {
    static constexpr size_t value = Idx;
};

template<typename Func, typename Last>
void for_each_impl(Func&& f, Last&& last) {
    f(last);
}

template<typename Func, typename First, typename ... Rest>
void for_each_impl(Func&& f, First&& first, Rest&&...rest) {
    f(first);
    for_each_impl( std::forward<Func>(f), rest...);
}

template <typename Func, int ... Indexes>
void for_each_N_helper(Func&& f, index_tuple<Indexes...>) {
    for_each_impl<Func>(std::forward<Func>(f), tuple_index<Indexes>() ...);
}

template <int I, typename IndexTuple> 
struct make_range_impl; 

template <int I, int... Indexes> 
struct make_range_impl<I, index_tuple<Indexes...>> {
    typedef typename make_range_impl<I - 1, index_tuple<Indexes..., I - 1>>::type type; 
};

template <int ... Indexes> 
struct make_range_impl<0, index_tuple<Indexes...>> {
    typedef index_tuple<Indexes...> type; 
};

template <typename Func, typename ... Args>
void for_tuple_range(const std::tuple<Args ...>&, Func&& f) {
    for_each_N_helper(
        std::forward<Func>(f),
        typename make_range_impl<
            std::tuple_size<std::tuple<Args ...>>::value,
            index_tuple<>>::type());
}

namespace std {
// From https://stackoverflow.com/questions/24609271/errormake-unique-is-not-a-member-of-std
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace std

#endif /* CUNVSM_CPP14_H */