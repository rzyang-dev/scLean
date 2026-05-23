#ifndef SCLEAN_DATA_SOURCE_H
#define SCLEAN_DATA_SOURCE_H

#include <string>
#include <vector>
#include <cstdint>
#include "sclean_types.h"

namespace sclean {

// Forward declarations
class DiskMatrix;
class HDF5File;

// Abstract data access layer.
// Operators request data by *logical name*, not HDF5 path.
// The HDF5DataSource implementation knows where each logical entity
// lives in the HDF5 file tree.
class DataSource {
public:
    virtual ~DataSource() = default;

    // -- Layer access (read CSC matrices from HDF5) --
    // "counts" -> /assays/<assay>/layers/counts
    // "data"   -> /assays/<assay>/layers/data
    virtual DiskMatrix* open_csc_layer(const std::string& name) = 0;

    // -- Feature vector access (1D datasets under /features/) --
    virtual std::vector<double> read_feature_vector(const std::string& name) = 0;
    virtual void write_feature_vector(const std::string& name,
                                      const std::vector<double>& data) = 0;

    // -- Int32 feature vectors (e.g., variable feature flags) --
    virtual std::vector<int32> read_feature_vector_int32(const std::string& name) = 0;
    virtual void write_feature_vector_int32(const std::string& name,
                                            const std::vector<int32>& data) = 0;

    // -- Cell/feature metadata --
    virtual std::vector<std::string> cell_names() = 0;
    virtual std::vector<std::string> feature_names() = 0;

    // -- Dimension queries --
    virtual int64 n_genes() const = 0;
    virtual int64 n_cells() const = 0;

    // -- HDF5 file access (for operators that need raw HDF5) --
    virtual HDF5File* file() = 0;
    virtual std::string assay_group() const = 0;
};

} // namespace sclean

#endif // SCLEAN_DATA_SOURCE_H
