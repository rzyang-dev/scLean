#include "hdf5_utils.h"
#include <stdexcept>
#include <cstdio>
#include <algorithm>

namespace sclean {
namespace hdf5 {

void check_write(herr_t status, const std::string& path) {
    if (status < 0) {
        char buf[512];
        snprintf(buf, sizeof(buf), "HDF5 write failed: %s", path.c_str());
        throw std::runtime_error(buf);
    }
}

hid_t open_dset(hid_t file_id, const std::string& path) {
    hid_t d = H5Dopen2(file_id, path.c_str(), H5P_DEFAULT);
    if (d < 0) throw std::runtime_error("Cannot open HDF5 dataset: " + path);
    return d;
}

hsize_t dset_len(hid_t dset) {
    hid_t space = H5Dget_space(dset);
    hsize_t len;
    H5Sget_simple_extent_dims(space, &len, nullptr);
    H5Sclose(space);
    return len;
}

hid_t create_1d(hid_t file_id, const std::string& path, hid_t dtype,
                hsize_t len, int compression) {
    // Create intermediate groups
    std::string group_path;
    size_t slash = 0;
    while ((slash = path.find('/', slash + 1)) != std::string::npos) {
        std::string g = path.substr(0, slash);
        if (H5Lexists(file_id, g.c_str(), H5P_DEFAULT) <= 0) {
            hid_t grp = H5Gcreate2(file_id, g.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(grp);
        }
    }

    if (H5Lexists(file_id, path.c_str(), H5P_DEFAULT) > 0) {
        H5Ldelete(file_id, path.c_str(), H5P_DEFAULT);
    }

    hsize_t chunk_dims[1] = {std::min(len, static_cast<hsize_t>(HDF5_CHUNK_SIZE_1D))};
    hid_t plist = H5P_DEFAULT;
    if (compression > 0 && len > chunk_dims[0]) {
        plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 1, chunk_dims);
        H5Pset_deflate(plist, compression);
    }

    hid_t space = H5Screate_simple(1, &len, nullptr);
    hid_t dset = H5Dcreate2(file_id, path.c_str(), dtype, space,
                             H5P_DEFAULT, plist, H5P_DEFAULT);
    H5Sclose(space);
    if (compression > 0 && len > chunk_dims[0]) H5Pclose(plist);
    return dset;
}

} // namespace hdf5
} // namespace sclean
