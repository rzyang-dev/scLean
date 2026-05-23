#ifndef SCLEAN_HDF5_UTILS_H
#define SCLEAN_HDF5_UTILS_H

#include <hdf5.h>
#include <string>
#include "scLean_types.h"

namespace sclean {
namespace hdf5 {

// Error-checking wrapper for HDF5 write operations.
// Throws std::runtime_error with the dataset path on failure.
void check_write(herr_t status, const std::string& path);

// Open an existing HDF5 dataset. Throws if the dataset cannot be opened.
hid_t open_dset(hid_t file_id, const std::string& path);

// Get the length (number of elements) of a 1D dataset.
hsize_t dset_len(hid_t dset);

// Create a 1D dataset with optional compression.
// Creates intermediate groups as needed. Deletes any existing dataset at path.
hid_t create_1d(hid_t file_id, const std::string& path, hid_t dtype,
                hsize_t len, int compression);

} // namespace hdf5
} // namespace sclean

#endif // SCLEAN_HDF5_UTILS_H
