#include <string>

#include "cuNVSM/data.h"
#include "cuNVSM/cuda_utils.h"

namespace TextEntity {

Batch::Batch(const size_t batch_size, const size_t window_size)
        : batch_size_(batch_size), window_size_(window_size),
          features_(nullptr), feature_weights_(nullptr), labels_(nullptr), weights_(nullptr),
          num_instances_(0) {
    CHECK_GT(batch_size_, 0);
    CHECK_GT(window_size_, 0);

    CCE(cudaHostAlloc(&features_,
                      batch_size_ * window_size_ * sizeof(FeaturesType),
                      cudaHostAllocDefault));
    CCE(cudaHostAlloc(&feature_weights_,
                      batch_size_ * window_size_ * sizeof(WeightType),
                      cudaHostAllocDefault));

    CCE(cudaHostAlloc(&labels_,
                      batch_size_ * sizeof(LabelsType),
                      cudaHostAllocDefault));
    CCE(cudaHostAlloc(&weights_,
                      batch_size_ * sizeof(WeightType),
                      cudaHostAllocDefault));

    clear();
}

Batch::Batch(const lse::TrainConfig& train_config)
        : Batch(train_config.batch_size(), train_config.window_size()) {}

Batch::Batch(Batch&& other)
        : batch_size_(other.batch_size_),
          window_size_(other.window_size_),
          features_(nullptr), feature_weights_(nullptr), labels_(nullptr), weights_(nullptr),
          num_instances_(0)  {
    swap(&other);
}

Batch::~Batch() {
    // This happens when the move constructor has been invoked.
    if (features_ == nullptr && feature_weights_ == nullptr && labels_ == nullptr && weights_ == nullptr) {
        return;
    }

    CCE(cudaFreeHost(features_));
    features_= nullptr;

    CCE(cudaFreeHost(feature_weights_));
    feature_weights_= nullptr;

    CCE(cudaFreeHost(labels_));
    labels_ = nullptr;

    CCE(cudaFreeHost(weights_));
    weights_= nullptr;
}

void Batch::clear() {
    num_instances_ = 0;
}

bool Batch::full() const {
    DCHECK_LE(num_instances_, batch_size_);

    return num_instances_ == batch_size_;
}

bool Batch::empty() const {
    return num_instances_ == 0;
}

void Batch::swap(BatchInterface* const other) {
    Batch* const other_batch = dynamic_cast<Batch*>(other);

    CHECK_NOTNULL(other_batch);

    CHECK_EQ(batch_size_, other_batch->batch_size_);
    CHECK_EQ(window_size_, other_batch->window_size_);

    // Swap data pointers.
    std::swap(features_, other_batch->features_);
    std::swap(feature_weights_, other_batch->feature_weights_);
    std::swap(labels_, other_batch->labels_);
    std::swap(weights_, other_batch->weights_);

    // Swap meta-data.
    std::swap(num_instances_, other_batch->num_instances_);
}

void DataSource::push_instance(
        const std::vector<WordIdxType>& features,
        const std::vector<FLOATING_POINT_TYPE> feature_weights,
        const ObjectIdxType object_id,
        const FLOATING_POINT_TYPE weight,
        Batch* const batch) {
    if (batch->full()) {
        overflow_buffer_.push_back(std::make_tuple(features, feature_weights, object_id, weight));
    } else {
        DCHECK_EQ(features.size(), batch->window_size());

        std::copy(features.begin(), features.end(),
                  &batch->features_[batch->num_instances_ * batch->window_size()]);

        if (!feature_weights.empty()) {
            CHECK_EQ(feature_weights.size(), features.size());

            std::copy(feature_weights.begin(), feature_weights.end(),
                      &batch->feature_weights_[batch->num_instances_ * batch->window_size()]);
        } else {
            std::fill(&batch->feature_weights_[batch->num_instances_ * batch->window_size()],
                      &batch->feature_weights_[(batch->num_instances_ + 1) * batch->window_size()],
                      static_cast<FLOATING_POINT_TYPE>(1.0));
        }

        batch->labels_[batch->num_instances_] = object_id;
        batch->weights_[batch->num_instances_] = weight;

        ++batch->num_instances_;
    }
}

std::ostream& operator<<(std::ostream& os, const Batch& batch) {
    os << "Window size: " << batch.window_size_ << std::endl;
    os << "Batch size: " << batch.batch_size_ << std::endl;
    os << "Number of instances: " << batch.num_instances_ << std::endl;

    os << "Features (" << batch.features_ << "): ";
    for (size_t i = 0; i < batch.num_instances_ * batch.window_size_; ++i)
        os << batch.features_[i] << " ";
    os << std::endl;

    os << "Labels (" << batch.labels_ << "): ";
    for (size_t i = 0; i < batch.num_instances_; ++i)
        os << batch.labels_[i] << " ";
    os << std::endl;

    os << "Weights (" << batch.weights_ << "): ";
    for (size_t i = 0; i < batch.num_instances_; ++i)
        os << batch.weights_[i] << " ";
    os << std::endl;

    return os;
}

}  // namespace TextEntity

namespace RepresentationSimilarity {

// TODO(cvangysel): some of the logic below is the same for both Batch types; merge?
Batch::Batch(const size_t batch_size)
        : batch_size_(batch_size), features_(nullptr), weights_(nullptr), num_instances_(0) {
    CHECK_GT(batch_size_, 0);

    CCE(cudaHostAlloc(&features_,
                      batch_size_ * 2 * sizeof(ObjectIdxType),
                      cudaHostAllocDefault));

    CCE(cudaHostAlloc(&weights_,
                      batch_size_ * sizeof(WeightType),
                      cudaHostAllocDefault));

    clear();
}

Batch::Batch(const lse::TrainConfig& train_config)
        : Batch(train_config.batch_size()) {}

Batch::Batch(Batch&& other)
        : batch_size_(other.batch_size_), features_(nullptr), weights_(nullptr), num_instances_(0)  {
    swap(&other);
}

Batch::~Batch() {
    // This happens when the move constructor has been invoked.
    if (features_ == nullptr && weights_ == nullptr) {
        return;
    }

    CCE(cudaFreeHost(features_));
    features_= nullptr;

    CCE(cudaFreeHost(weights_));
    weights_ = nullptr;
}

void Batch::clear() {
    num_instances_ = 0;
}

bool Batch::full() const {
    DCHECK_LE(num_instances_, batch_size_);

    return num_instances_ == batch_size_;
}

void Batch::swap(BatchInterface* const other) {
    Batch* const other_batch = dynamic_cast<Batch*>(other);

    CHECK_NOTNULL(other_batch);

    CHECK_EQ(batch_size_, other_batch->batch_size_);

    // Swap data pointers.
    std::swap(features_, other_batch->features_);
    std::swap(weights_, other_batch->weights_);

    // Swap meta-data.
    std::swap(num_instances_, other_batch->num_instances_);
}

bool Batch::empty() const {
    return num_instances_ == 0;
}

std::ostream& operator<<(std::ostream& os, const Batch& batch) {
    os << "Batch size: " << batch.batch_size_ << std::endl;
    os << "Number of instances: " << batch.num_instances_ << std::endl;

    os << "Features (" << batch.features_ << "): ";
    for (size_t i = 0; i < batch.num_instances_ * 2; ++i)
        os << batch.features_[i] << " ";
    os << "Weights (" << batch.weights_ << "): ";
    for (size_t i = 0; i < batch.num_instances_; ++i)
        os << batch.weights_[i] << " ";
    os << std::endl;

    return os;
}

std::vector<InstanceT>* LoadSimilarities(
        std::istream& file,
        const IdentifiersMapT& identifiers_map) {
    CHECK(file.good());
    CHECK(!identifiers_map.empty());

    std::vector<InstanceT>* const data = new std::vector<InstanceT>;

    std::string line;

    while (file.good() && std::getline(file, line)) {
        std::istringstream iss(line);

        std::string first_entity_id;
        std::string second_entity_id;
        WeightType weight;

        iss >> first_entity_id;
        iss >> second_entity_id;
        iss >> weight;

        if (!contains_key(identifiers_map, first_entity_id)) {
            LOG(WARNING) << "Entity '" << first_entity_id << "' not found; "
                         << "skipping pair.";

            continue;
        }

        if (!contains_key(identifiers_map, second_entity_id)) {
            LOG(WARNING) << "Entity '" << second_entity_id << "' not found; "
                         << "skipping pair.";

            continue;
        }

        data->push_back(std::make_tuple(
            identifiers_map.at(first_entity_id),
            identifiers_map.at(second_entity_id),
            weight));
    }

    return data;
}

std::vector<InstanceT>* LoadSimilarities(
        const std::string& path,
        const IdentifiersMapT& identifiers_map) {
    CHECK(!path.empty());

    std::ifstream file;
    file.open(path);

    return LoadSimilarities(file, identifiers_map);
}

DataSource::DataSource(const std::string& path,
                       const IdentifiersMapT& identifiers_map,
                       RNG* const rng)
        : DataSource(LoadSimilarities(path, identifiers_map), rng) {}

DataSource::DataSource(const std::vector<InstanceT>* const data,
                       RNG* const rng)
        : data_(data), rng_(rng) {
    reset();
}

void DataSource::reset() {
    if (!instance_order_.empty()) {
        LOG(WARNING) << "Resetting instance generator while there are still instances to consume.";

        instance_order_.clear();
    }

    instance_order_.resize(data_->size());

    // Fill instance order with 0, ..., n-1.
    std::iota(std::begin(instance_order_), std::end(instance_order_), 0);

    LOG(INFO) << "Shuffling " << instance_order_.size() << " instance pointers.";
    std::shuffle(instance_order_.begin(), instance_order_.end(), *rng_);
}

void DataSource::next(Batch* const batch) {
    CHECK(batch->empty());

    while (!batch->full() && !instance_order_.empty()) {
        const InstanceT& instance = data_->at(instance_order_.front());

        const size_t offset = 2 * batch->num_instances_;
        batch->features_[offset] = std::get<0>(instance);
        batch->features_[offset + 1] = std::get<1>(instance);

        batch->weights_[batch->num_instances_] = std::get<2>(instance);

        instance_order_.pop_front();

        ++batch->num_instances_;
    }

    CHECK_LE(batch->num_instances_, batch->batch_size_);
}

bool DataSource::has_next() const {
    return !instance_order_.empty();
}

float32 DataSource::progress() const {
    return 1.0 - (
        static_cast<float32>(instance_order_.size()) /
        static_cast<float32>(data_->size()));
}

}  // namespace RepresentationSimilarity
