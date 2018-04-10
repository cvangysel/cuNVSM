#ifndef CUNVSM_HDF5_H
#define CUNVSM_HDF5_H

#include <H5Cpp.h>
#include <H5Classes.h>

#include <glog/logging.h>

#include "cuNVSM/model.h"
#include "cuNVSM/cuda_utils.h"

void write_to_hdf5(const std::string& name,
                   const device_matrix<typename LSE::FloatT>& matrix,
                   H5::H5File* const file);

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
void write_to_hdf5(const ModelBase<FloatT, WordIdxType, EntityIdxType>& model,
                   H5::H5File* const file);

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
void write_to_hdf5(const ModelBase<FloatT, WordIdxType, EntityIdxType>& model,
                   const std::string& filename);

#include "lse_hdf5_inl.h"

#endif /* CUNVSM_HDF5_H */