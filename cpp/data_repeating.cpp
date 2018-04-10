#include "cuNVSM/data.h"

template <typename BatchT>
RepeatingSource<BatchT>::RepeatingSource(const size_t num_repeats,
                                         DataSource<BatchT>* source)
        : source_(source), num_repeats_(num_repeats),
          current_iteration_(0) {}

template <typename BatchT>
void RepeatingSource<BatchT>::reset() {
    current_iteration_ = 0;
    source_->reset();
}

template <typename BatchT>
void RepeatingSource<BatchT>::next(BatchType* const batch) {
    if (!source_->has_next()) {
        source_->reset();
        ++ current_iteration_;

        CHECK_LT(current_iteration_, num_repeats_);
    }

    source_->next(batch);
}

template <typename BatchT>
bool RepeatingSource<BatchT>::has_next() const {
    if (current_iteration_ + 1 < num_repeats_) {
        return true;
    } else if (current_iteration_ + 1 == num_repeats_) {
        return source_->has_next();
    }

    LOG(FATAL) << "This should not happen.";
    throw 0;
}

template <typename BatchT>
float32 RepeatingSource<BatchT>::progress() const {
    return source_->progress() + (
        static_cast<float32>(current_iteration_) / num_repeats_);
}

template <typename BatchT>
void RepeatingSource<BatchT>::extract_metadata(
        lse::Metadata* const metadata) const {
    source_->extract_metadata(metadata);
}

// Explicit instantiations.
template class RepeatingSource<TextEntity::Batch>;
template class RepeatingSource<RepresentationSimilarity::Batch>;
