#include "cuNVSM/labels.h"

template <typename FloatT, typename EntityIdxType>
void UniformLabelGenerator<FloatT, EntityIdxType>::generate(
        const EntityIdxType* const labels,
        const Representations<FloatT, EntityIdxType>& entities,
        const size_t num_labels,
        const size_t num_negative_labels,
        std::vector<EntityIdxType>* const instance_entities,
        RNG* const rng) const {
    const size_t num_repeats = num_negative_labels + 1;

    for (size_t idx = 0; idx < num_labels; ++idx) {
        (*instance_entities)[idx * num_repeats] = labels[idx];

        generate_random_indexes<EntityIdxType>(
            entities.num_objects(),
            num_negative_labels,
            rng,
            instance_entities->data() + idx * num_repeats + 1);
    }
}

// Explicit instantiations.
template class UniformLabelGenerator<FLOATING_POINT_TYPE, int32>;