#include <hdf5.h>
#include <H5Cpp.h>
#include <H5Classes.h>
#include <H5PredType.h>

#include "cuNVSM/base.h"
#include "cuNVSM/cuda_utils.h"
#include "cuNVSM/hdf5.h"
#include "cuNVSM/model.h"

template <typename FloatT>
struct GetH5PredType {};
template <>
struct GetH5PredType<float32> {
    static H5::PredType type() {
        return H5::PredType::NATIVE_FLOAT;
    }
};
template <>
struct GetH5PredType<float64> {
    static H5::PredType type() {
        return H5::PredType::NATIVE_DOUBLE;
    }
};

void write_to_hdf5(const std::string& name,
                   const device_matrix<typename LSE::FloatT>& matrix,
                   H5::H5File* const file) {
    PROFILE_FUNCTION();

    CHECK_NOTNULL(file);

    hsize_t dim[] = {matrix.getCols(), matrix.getRows()};
    H5::DataSpace dataspace(2, /* rank */
                            dim);

    VLOG(3) << "Data dimensionality: " << dim << ".";

    H5::FloatType datatype(GetH5PredType<LSE::FloatT>::type());
    datatype.setOrder(H5T_ORDER_LE);

    H5::DataSet dataset = file->createDataSet(name, datatype, dataspace);

    VLOG(3) << "Fetching data from device.";

    // Transfer data from device to host.
    LSE::FloatT* const data = get_array(matrix.getStream(), matrix);

    VLOG(3) << "Writing data to HDF5.";

    dataset.write(data, GetH5PredType<LSE::FloatT>::type());
    delete [] data;
}
