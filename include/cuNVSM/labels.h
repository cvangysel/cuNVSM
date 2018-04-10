#ifndef CUNVSM_LABELS_H
#define CUNVSM_LABELS_H

#include "cuNVSM/base.h"
#include "cuNVSM/params.h"

template <typename FloatT, typename EntityIdxType>
class LabelGenerator {
 public:
  virtual ~LabelGenerator() {}

  virtual void generate(const EntityIdxType* const labels,
                        const Representations<FloatT, EntityIdxType>& entities,
                        const size_t num_labels,
                        const size_t num_negative_labels,
                        std::vector<EntityIdxType>* const instance_entities,
                        RNG* const rng) const = 0;
};

template <typename FloatT, typename EntityIdxType>
class UniformLabelGenerator : public LabelGenerator<FloatT, EntityIdxType> {
 public:
  virtual void generate(const EntityIdxType* const labels,
                        const Representations<FloatT, EntityIdxType>& entities,
                        const size_t num_labels,
                        const size_t num_negative_labels,
                        std::vector<EntityIdxType>* const instance_entities,
                        RNG* const rng) const override;
};

#endif /* CUNVSM_LABELS_H */