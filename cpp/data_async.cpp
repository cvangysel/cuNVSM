#include "cuNVSM/data.h"

template <typename BatchType>
void async_worker(DataSource<BatchType>* const source,
                  std::atomic_bool* const is_running,
                  boost::lockfree::queue<BatchType*>* const input,
                  boost::lockfree::queue<BatchType*>* const output) {
    VLOG(2) << "Started data-fetching worker.";

    CHECK(!is_running->load());
    is_running->store(true);

    while (source->has_next() && is_running->load()) {
        if (input->empty()) {
            std::this_thread::yield();

            continue;
        }

        // Load batch from input.
        BatchType* batch;
        input->pop(batch);

        // Populate batch.
        source->next(batch);

        // Release.
        output->push(batch);
    }

    is_running->store(false);

    VLOG(2) << "Data-fetching worker finished.";
}

template <typename BatchT>
AsyncSource<BatchT>::AsyncSource(
    const size_t num_concurrent_batches,
    const size_t batch_size,
    const size_t window_size,
    DataSource<BatchType>* const source)
        : num_concurrent_batches_(num_concurrent_batches),
          source_(source),
          buffers_(num_concurrent_batches),
          empty_batches_(num_concurrent_batches),
          full_batches_(num_concurrent_batches),
          thread_(nullptr),
          is_running_(false),
          finished_() {
    for (auto& buffer : buffers_) {
        buffer.reset(new BatchT(batch_size, window_size));

        // Put into queue.
        empty_batches_.push(buffer.get());
    }

    start_worker();
}

template <typename BatchT>
AsyncSource<BatchT>::~AsyncSource() {
    stop_worker();
}

template <typename BatchT>
void AsyncSource<BatchT>::reset() {
    source_->reset();

    stop_worker();
    start_worker();
}

template <typename BatchT>
void AsyncSource<BatchT>::next(BatchType* const batch) {
    // TODO(cvangysel): figure out why this call was here in the first place.
    // DataSource::next(batch);

    // Make sure the internals did not add something to the batch.
    DCHECK(batch->empty());

    // Pre-condition: make sure there is still more.
    CHECK(has_next());

    while (!batches_ready()) {
        std::this_thread::yield();
    }

    BatchType* buffer_batch;
    full_batches_.pop(buffer_batch);

    batch->swap(buffer_batch);

    // Make sure the swap was good.
    DCHECK(buffer_batch->empty());
    DCHECK(!batch->empty());

    // Release buffer back into the wild.
    empty_batches_.push(buffer_batch);
}

template <typename BatchT>
bool AsyncSource<BatchT>::has_next() const {
    // Easy case; proxy.
    if (source_->has_next()) {
        CHECK(worker_running());

        return true;
    }

    // As long as the worker has not finished, see if there are any batches.
    while (worker_running()) {
        if (batches_ready()) {
            return true;
        }

        std::this_thread::yield();
    }

    if (batches_ready()) {
        return true;
    }

    return false;
}

template <typename BatchT>
float32 AsyncSource<BatchT>::progress() const {
    // Proxy.
    return source_->progress();
}

template <typename BatchT>
void AsyncSource<BatchT>::extract_metadata(lse::Metadata* const metadata) const {
    // Proxy.
    return source_->extract_metadata(metadata);
}


template <typename BatchT>
bool AsyncSource<BatchT>::batches_ready() const {
    // Hack to fix Boost library incompatibilities.
    return !const_cast<AsyncSource*>(this)->full_batches_.empty();
}

template <typename BatchT>
bool AsyncSource<BatchT>::worker_running() const {
    return thread_.get() != nullptr &&
        finished_.wait_for(std::chrono::seconds(0)) !=
            std::future_status::ready;
}

template <typename BatchT>
void AsyncSource<BatchT>::stop_worker() {
    if (!worker_running()) {
        return;
    }

    while (worker_running() && !is_running_.load()) {
        std::this_thread::yield();
    }

    is_running_.store(false);

    if (thread_->joinable()) {
        thread_->join();
    }

    thread_.reset(nullptr);
}

template <typename BatchT>
void AsyncSource<BatchT>::start_worker() {
    CHECK(!worker_running());

      std::packaged_task<void(DataSource<BatchType>* const,
                              std::atomic_bool* const,
                              result_queue* const,
                              result_queue* const)> task(async_worker<BatchType>);

      finished_ = task.get_future();

      if ((thread_.get() != nullptr) && (thread_->joinable())) {
          thread_->join();
      }

    thread_.reset(new std::thread(std::move(task),
                                  source_.get(),
                                  &is_running_,
                                  &empty_batches_,
                                  &full_batches_));
}

// Explicit instantiations.
template class AsyncSource<TextEntity::Batch>;
template class AsyncSource<RepresentationSimilarity::Batch>;
