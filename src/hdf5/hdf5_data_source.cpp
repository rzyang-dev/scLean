#include "hdf5/hdf5_data_source.h"
#include "hdf5/hdf5_csc_matrix.h"
#include "hdf5/hdf5_file.h"

namespace sclean {

DiskMatrix* HDF5DataSource::open_csc_layer(const std::string& name) {
    return file_->open_csc_matrix(layer_path(name)).release();
}

std::vector<double> HDF5DataSource::read_feature_vector(const std::string& name) {
    return file_->read_vector_double(feature_vector_path(name));
}

void HDF5DataSource::write_feature_vector(const std::string& name,
                                          const std::vector<double>& data) {
    file_->write_vector_double(feature_vector_path(name), data);
}

std::vector<int32> HDF5DataSource::read_feature_vector_int32(const std::string& name) {
    return file_->read_vector_int32(feature_vector_path(name));
}

void HDF5DataSource::write_feature_vector_int32(const std::string& name,
                                                const std::vector<int32>& data) {
    file_->write_vector_int32(feature_vector_path(name), data);
}

std::vector<std::string> HDF5DataSource::cell_names() {
    return file_->read_vector_string("/cells/names");
}

std::vector<std::string> HDF5DataSource::feature_names() {
    return file_->read_vector_string("/features/names");
}

int64 HDF5DataSource::n_genes() const {
    if (cached_n_genes_ < 0) {
        auto mat = file_->open_csc_matrix(layer_path("data"));
        cached_n_genes_ = mat->n_rows();
    }
    return cached_n_genes_;
}

int64 HDF5DataSource::n_cells() const {
    if (cached_n_cells_ < 0) {
        auto mat = file_->open_csc_matrix(layer_path("data"));
        cached_n_cells_ = mat->n_cols();
    }
    return cached_n_cells_;
}

} // namespace sclean
