#ifndef SCLEAN_HDF5_DATA_SOURCE_H
#define SCLEAN_HDF5_DATA_SOURCE_H

#include "core/data_source.h"
#include <string>

namespace sclean {

// Concrete DataSource backed by an HDF5 file and assay group.
// Centralizes all HDF5 path knowledge so operators request data
// by logical name rather than hardcoded filesystem paths.
class HDF5DataSource : public DataSource {
public:
    HDF5DataSource(HDF5File* file, const std::string& assay_group)
        : file_(file), assay_group_(assay_group) {}

    // -- DataSource interface --
    DiskMatrix* open_csc_layer(const std::string& name) override;

    std::vector<double> read_feature_vector(const std::string& name) override;
    void write_feature_vector(const std::string& name,
                              const std::vector<double>& data) override;

    std::vector<int32> read_feature_vector_int32(const std::string& name) override;
    void write_feature_vector_int32(const std::string& name,
                                    const std::vector<int32>& data) override;

    std::vector<std::string> cell_names() override;
    std::vector<std::string> feature_names() override;

    int64 n_genes() const override;
    int64 n_cells() const override;

    HDF5File* file() override { return file_; }
    std::string assay_group() const override { return assay_group_; }

    // -- Path-construction helpers (inline) — single source of truth for HDF5 layout --

    std::string layer_path(const std::string& name) const {
        return assay_group_ + "/layers/" + name;
    }
    std::string layer_subpath(const std::string& layer, const std::string& sub) const {
        return assay_group_ + "/layers/" + layer + "/" + sub;
    }
    std::string feature_vector_path(const std::string& name) const {
        return assay_group_ + "/features/" + name;
    }
    std::string graph_path(const std::string& name) const {
        return assay_group_ + "/graphs/" + name;
    }
    std::string reduction_path(const std::string& name) const {
        return assay_group_ + "/reductions/" + name;
    }
    std::string reduction_subpath(const std::string& reduction, const std::string& sub) const {
        return assay_group_ + "/reductions/" + reduction + "/" + sub;
    }
    std::string cluster_path(const std::string& name) const {
        return assay_group_ + "/clusters/" + name;
    }

private:
    HDF5File* file_;
    std::string assay_group_;
    mutable int64 cached_n_genes_ = -1;   // lazy, from open_csc_layer("data")
    mutable int64 cached_n_cells_ = -1;
};

} // namespace sclean

#endif // SCLEAN_HDF5_DATA_SOURCE_H
