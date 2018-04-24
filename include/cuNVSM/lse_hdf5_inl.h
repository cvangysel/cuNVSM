#ifndef CUNVSM_HDF5_INL_H
#define CUNVSM_HDF5_INL_H

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
void write_to_hdf5(const ModelBase<FloatT, WordIdxType, EntityIdxType>& model,
                   H5::H5File* const file) {
    CHECK_NOTNULL(file);

    for (const auto& pair : model.get_data()) {
        const std::string& name = pair.first;
        const device_matrix<DefaultModel::FloatT>* const matrix = pair.second;

        CHECK_NOTNULL(matrix);

        VLOG(2) << "Writing " << name << " (" << *matrix << ") "
                << "to " << file << ".";

        write_to_hdf5(name, *matrix, file);
    }
}

template <typename FloatT, typename WordIdxType, typename EntityIdxType>
void write_to_hdf5(const ModelBase<FloatT, WordIdxType, EntityIdxType>& model,
                   const std::string& filename) {
    H5::H5File file(filename, H5F_ACC_EXCL /* fail if file already exists */);
    write_to_hdf5(model, &file);
}

#endif /* CUNVSM_HDF5_INL_H */