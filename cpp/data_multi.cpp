#include "cuNVSM/data.h"

template <typename ... BatchT>
class ConstructorFn {
 public:
  ConstructorFn(const std::tuple<DataSource<BatchT>* ...>& src,
                std::tuple<std::unique_ptr<DataSource<BatchT>> ...>& dst)
      : src_(&src), dst_(&dst) {}

  template <typename Index>
  void operator()(const Index& idx) {
      std::get<Index::value>(*dst_).reset(std::get<Index::value>(*src_));
  }

 private:
  const std::tuple<DataSource<BatchT>* ...>* const src_;
  std::tuple<std::unique_ptr<DataSource<BatchT>> ...>* const dst_;

  DISALLOW_COPY_AND_ASSIGN(ConstructorFn);
};

template <typename ... BatchT>
MultiSource<BatchT ...>::MultiSource(
        const std::tuple<DataSource<BatchT>* ...>& sources) : sources_() {
    for_tuple_range(sources_, ConstructorFn<BatchT ...>(sources, sources_));
}

template <typename ... BatchT>
MultiSource<BatchT ...>::~MultiSource() {}

template <typename ... BatchT>
class ResetFn {
 public:
  ResetFn(std::tuple<std::unique_ptr<DataSource<BatchT>> ...>& sources)
      : sources_(&sources) {}

  template <typename Index>
  void operator()(const Index& idx) {
      std::get<Index::value>(*sources_)->reset();
  }

 private:
  std::tuple<std::unique_ptr<DataSource<BatchT>> ...>* sources_;

  DISALLOW_COPY_AND_ASSIGN(ResetFn);
};

template <typename ... BatchT>
void MultiSource<BatchT ...>::reset() {
    for_tuple_range(sources_, ResetFn<BatchT ...>(sources_));
}

template <typename ... BatchT>
class NextFn {
 public:
  NextFn(std::tuple<std::unique_ptr<DataSource<BatchT>> ...>& sources,
         std::tuple<BatchT ...>* const batch)
      : sources_(&sources), batch_(batch) {}

  template <typename Index>
  void operator()(const Index& idx) {
      std::get<Index::value>(*sources_)->next(&std::get<Index::value>(*batch_));
  }

 private:
  std::tuple<std::unique_ptr<DataSource<BatchT>> ...>* const sources_;
  std::tuple<BatchT ...>* const batch_;

  DISALLOW_COPY_AND_ASSIGN(NextFn);
};

template <typename ... BatchT>
void MultiSource<BatchT ...>::next(BatchType* const batch) {
    for_tuple_range(sources_, NextFn<BatchT ...>(sources_, batch));
}

template <typename ReturnT, typename ... BatchT>
class AggregatorFn {
 public:
  AggregatorFn(const ReturnT initial_value,
               const std::tuple<std::unique_ptr<DataSource<BatchT>> ...>& sources)
      : value_(initial_value), sources_(&sources) {}

  template <typename Index>
  void operator()(const Index& idx) {
      this->perform(std::get<Index::value>(*sources_).get());
  }

  ReturnT get_value() const {
      return value_;
  }

 protected:
  virtual void perform(DataSourceInterface* const source) = 0;

  ReturnT value_;

 private:
  const std::tuple<std::unique_ptr<DataSource<BatchT>> ...>* const sources_;
};

template <typename ... BatchT>
class HasNextFn : public AggregatorFn<bool, BatchT ...> {
 public:
  HasNextFn(const std::tuple<std::unique_ptr<DataSource<BatchT>> ...>& sources)
      : AggregatorFn<bool, BatchT ...>(true, sources) {}

 protected:
  virtual void perform(DataSourceInterface* const source) override {
      this->value_ = this->value_ && source->has_next();
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(HasNextFn);
};

template <typename ... BatchT>
bool MultiSource<BatchT ...>::has_next() const {
    HasNextFn<BatchT ...> fn(sources_);
    for_tuple_range(sources_, fn);

    return fn.get_value();
}

template <typename ... BatchT>
class ProgressFn : public AggregatorFn<float32, BatchT ...> {
 public:
  ProgressFn(const std::tuple<std::unique_ptr<DataSource<BatchT>> ...>& sources)
      : AggregatorFn<float32, BatchT ...>(1.0f, sources) {}

 protected:
  virtual void perform(DataSourceInterface* const source) override {
      this->value_ = std::min(this->value_, source->progress());
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(ProgressFn);
};

template <typename ... BatchT>
float32 MultiSource<BatchT ...>::progress() const {
    ProgressFn<BatchT ...> fn(sources_);
    for_tuple_range(sources_, fn);

    return fn.get_value();
}

template <typename ... BatchT>
class ExtractMetadataFn : public AggregatorFn<void*, BatchT ...> {
 public:
  ExtractMetadataFn(const std::tuple<std::unique_ptr<DataSource<BatchT>> ...>& sources,
                    lse::Metadata* const metadata)
      : AggregatorFn<void*, BatchT ...>((void*) nullptr, sources),
        metadata_(metadata) {}

 protected:
  virtual void perform(DataSourceInterface* const source) override {
      source->extract_metadata(metadata_);
  }

 private:
  lse::Metadata* const metadata_;

  DISALLOW_COPY_AND_ASSIGN(ExtractMetadataFn);
};

template <typename ... BatchT>
void MultiSource<BatchT ...>::extract_metadata(lse::Metadata* const metadata) const {
    for_tuple_range(sources_, ExtractMetadataFn<BatchT ...>(sources_, metadata));
}

// Explicit instantiations.
template class MultiSource<TextEntity::Batch, TextEntity::Batch>;  // For testing.
template class MultiSource<TextEntity::Batch, RepresentationSimilarity::Batch>;
