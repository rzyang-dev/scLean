#ifndef SCLEAN_HDF5_FILE_H
#define SCLEAN_HDF5_FILE_H

#include <hdf5.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "scLean_types.h"

namespace sclean {

enum class FileMode { ReadOnly, ReadWrite, Create };

class HDF5CSCMatrix;
class HDF5DenseMatrix;

class HDF5File {
public:
    explicit HDF5File(const std::string& path, FileMode mode = FileMode::ReadWrite);
    ~HDF5File();

    HDF5File(const HDF5File&) = delete;
    HDF5File& operator=(const HDF5File&) = delete;
    HDF5File(HDF5File&& other) noexcept;
    HDF5File& operator=(HDF5File&& other) noexcept;

    // Per-thread file handle for OpenMP parallel reads
    hid_t open_thread_handle(FileMode mode);
    void close_thread_handles();

    // Open matrices from groups/datasets
    std::unique_ptr<HDF5CSCMatrix> open_csc_matrix(const std::string& group_path,
                                                   hid_t thread_file = -1);
    std::unique_ptr<HDF5DenseMatrix> open_dense_matrix(const std::string& dataset_path);
    std::unique_ptr<HDF5DenseMatrix> create_dense_dataset(
        const std::string& path, int64 n_rows, int64 n_cols);

    // Create CSC matrix
    std::unique_ptr<HDF5CSCMatrix> create_csc_matrix(
        const std::string& group_path,
        const std::vector<double>& data,
        const std::vector<int32>& indices,
        const std::vector<int64>& indptr,
        int64 n_rows, int64 n_cols);

    // Existence / groups
    bool exists(const std::string& path) const;
    void create_group(const std::string& path);
    void delete_group(const std::string& path);

    // Attribute helpers
    void set_attr(const std::string& path, const std::string& name, int64 value);
    void set_attr_string(const std::string& path, const std::string& name,
                          const std::string& value);
    void set_attr_double(const std::string& path, const std::string& name, double value);
    int64 get_attr_int64(const std::string& path, const std::string& name);
    double get_attr_double(const std::string& path, const std::string& name);
    std::string get_attr_string(const std::string& path, const std::string& name);

    // 1D dataset read/write
    void write_vector_double(const std::string& path, const std::vector<double>& data,
                              int compression = 3);
    void write_vector_int32(const std::string& path, const std::vector<int32>& data);
    void write_vector_int64(const std::string& path, const std::vector<int64>& data);
    void write_vector_string(const std::string& path, const std::vector<std::string>& data);
    std::vector<double> read_vector_double(const std::string& path);
    std::vector<int32> read_vector_int32(const std::string& path);
    std::vector<int64> read_vector_int64(const std::string& path);
    std::vector<std::string> read_vector_string(const std::string& path);

    void flush();
    const std::string& path() const { return path_; }
    hid_t file_id() const { return file_id_; }
    FileMode mode() const { return mode_; }

    // Thread-safe write mutex (shared across all files of this assay)
    std::mutex write_mutex_;

private:
    std::string path_;
    hid_t file_id_;
    FileMode mode_;

    struct ThreadHandle {
        hid_t id;
        FileMode mode;
    };
    std::vector<ThreadHandle> thread_handles_;
    std::mutex handle_mutex_;

    static hid_t h5_open(const std::string& path, FileMode mode);
    void close_handle(hid_t h);
};

} // namespace sclean

#endif // SCLEAN_HDF5_FILE_H
