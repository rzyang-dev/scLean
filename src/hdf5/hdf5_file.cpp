#include "hdf5_file.h"
#include <R_ext/Print.h>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <hdf5_hl.h>
#include <unistd.h>

namespace sclean {

static void check_write(herr_t status, const char* op, const char* path) {
    if (status < 0) {
        char buf[512];
        snprintf(buf, sizeof(buf), "HDF5 %s failed: %s", op, path);
        throw std::runtime_error(buf);
    }
}

static void warn_obj_open(const char* path) {
    REprintf("[scLean] WARNING: H5Oopen failed for %s\n", path);
}

static const char* file_mode_to_h5_flags(FileMode mode) {
    switch (mode) {
        case FileMode::ReadOnly:  return "H5F_ACC_RDONLY";
        case FileMode::ReadWrite: return "H5F_ACC_RDWR";
        case FileMode::Create:    return "H5F_ACC_TRUNC";
    }
    return "H5F_ACC_RDONLY";
}

static unsigned file_mode_to_flags(FileMode mode) {
    switch (mode) {
        case FileMode::ReadOnly:  return H5F_ACC_RDONLY;
        case FileMode::ReadWrite: return H5F_ACC_RDWR;
        case FileMode::Create:    return H5F_ACC_TRUNC;
    }
    return H5F_ACC_RDONLY;
}

hid_t HDF5File::h5_open(const std::string& path, FileMode mode) {
    unsigned flags = file_mode_to_flags(mode);
    hid_t fapl = H5P_DEFAULT;

    // If ReadWrite and the file doesn't exist on disk, create it
    if (mode == FileMode::ReadWrite && access(path.c_str(), F_OK) != 0) {
        mode = FileMode::Create;
        flags = H5F_ACC_TRUNC;
    }

    if (mode == FileMode::ReadWrite || mode == FileMode::ReadOnly) {
        if (H5Fis_hdf5(path.c_str()) <= 0) {
            throw std::runtime_error("File is not a valid HDF5 file: " + path);
        }
    }

    hid_t id;
    if (mode == FileMode::Create) {
        id = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    } else {
        id = H5Fopen(path.c_str(), flags, fapl);
    }

    if (id < 0) {
        throw std::runtime_error("Failed to open HDF5 file: " + path);
    }
    return id;
}

HDF5File::HDF5File(const std::string& path, FileMode mode)
    : path_(path), mode_(mode) {
    file_id_ = h5_open(path, mode);
}

HDF5File::~HDF5File() {
    close_thread_handles();
    if (file_id_ >= 0) {
        H5Fclose(file_id_);
    }
}

HDF5File::HDF5File(HDF5File&& other) noexcept
    : path_(std::move(other.path_))
    , file_id_(other.file_id_)
    , mode_(other.mode_)
    , thread_handles_(std::move(other.thread_handles_)) {
    other.file_id_ = -1;
}

HDF5File& HDF5File::operator=(HDF5File&& other) noexcept {
    if (this != &other) {
        close_thread_handles();
        if (file_id_ >= 0) H5Fclose(file_id_);
        path_ = std::move(other.path_);
        file_id_ = other.file_id_;
        mode_ = other.mode_;
        thread_handles_ = std::move(other.thread_handles_);
        other.file_id_ = -1;
    }
    return *this;
}

hid_t HDF5File::open_thread_handle(FileMode mode) {
    // Read-only threads can share handles; writes need exclusive access
    if (mode == FileMode::ReadOnly) {
        static thread_local std::unordered_map<std::string, hid_t> cache;
        std::string key = path_ + ":r";
        auto it = cache.find(key);
        if (it != cache.end() && it->second >= 0) return it->second;

        hid_t h = h5_open(path_, FileMode::ReadOnly);
        cache[key] = h;
        std::lock_guard<std::mutex> lock(handle_mutex_);
        thread_handles_.push_back({h, FileMode::ReadOnly});
        return h;
    }
    return file_id_;
}

void HDF5File::close_thread_handles() {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    for (auto& th : thread_handles_) {
        if (th.id >= 0 && th.id != file_id_) {
            H5Fclose(th.id);
        }
    }
    thread_handles_.clear();
}

void HDF5File::close_handle(hid_t h) {
    if (h >= 0 && h != file_id_) {
        H5Fclose(h);
    }
}

bool HDF5File::exists(const std::string& path) const {
    return H5Lexists(file_id_, path.c_str(), H5P_DEFAULT) > 0;
}

void HDF5File::create_group(const std::string& path) {
    if (exists(path)) return;

    // Create intermediate groups
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        std::string g = path.substr(0, pos);
        if (!g.empty() && H5Lexists(file_id_, g.c_str(), H5P_DEFAULT) <= 0) {
            hid_t grp = H5Gcreate2(file_id_, g.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(grp);
        }
    }

    hid_t gcpl = H5P_DEFAULT;
    hid_t group = H5Gcreate2(file_id_, path.c_str(), H5P_DEFAULT, gcpl, H5P_DEFAULT);
    if (group < 0) {
        throw std::runtime_error("Failed to create HDF5 group: " + path);
    }
    H5Gclose(group);
}

void HDF5File::delete_group(const std::string& path) {
    if (exists(path)) {
        H5Ldelete(file_id_, path.c_str(), H5P_DEFAULT);
    }
}

void HDF5File::flush() {
    H5Fflush(file_id_, H5F_SCOPE_GLOBAL);
}

// --- Attributes ---

void HDF5File::set_attr(const std::string& path, const std::string& name, int64 value) {
    hid_t obj = H5Oopen(file_id_, path.c_str(), H5P_DEFAULT);
    if (obj < 0) { warn_obj_open(path.c_str()); return; }
    hsize_t dims[1] = {1};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t attr = H5Acreate2(obj, name.c_str(), H5T_NATIVE_INT64, space,
                             H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Awrite(attr, H5T_NATIVE_INT64, &value);
    check_write(status, "H5Awrite", (path + "/@" + name).c_str());
    H5Aclose(attr);
    H5Sclose(space);
    H5Oclose(obj);
}

void HDF5File::set_attr_double(const std::string& path, const std::string& name, double value) {
    hid_t obj = H5Oopen(file_id_, path.c_str(), H5P_DEFAULT);
    if (obj < 0) { warn_obj_open(path.c_str()); return; }
    hsize_t dims[1] = {1};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t attr = H5Acreate2(obj, name.c_str(), H5T_NATIVE_DOUBLE, space,
                             H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
    check_write(status, "H5Awrite", (path + "/@" + name).c_str());
    H5Aclose(attr);
    H5Sclose(space);
    H5Oclose(obj);
}

void HDF5File::set_attr_string(const std::string& path, const std::string& name,
                                const std::string& value) {
    hid_t obj = H5Oopen(file_id_, path.c_str(), H5P_DEFAULT);
    if (obj < 0) { warn_obj_open(path.c_str()); return; }
    hid_t strtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(strtype, value.size() + 1);
    hsize_t dims[1] = {1};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t attr = H5Acreate2(obj, name.c_str(), strtype, space, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Awrite(attr, strtype, value.c_str());
    check_write(status, "H5Awrite", (path + "/@" + name).c_str());
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(strtype);
    H5Oclose(obj);
}

int64 HDF5File::get_attr_int64(const std::string& path, const std::string& name) {
    hid_t obj = H5Oopen(file_id_, path.c_str(), H5P_DEFAULT);
    if (obj < 0) return 0;
    int64 val = 0;
    if (H5Aexists(obj, name.c_str()) > 0) {
        hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
        H5Aread(attr, H5T_NATIVE_INT64, &val);
        H5Aclose(attr);
    }
    H5Oclose(obj);
    return val;
}

double HDF5File::get_attr_double(const std::string& path, const std::string& name) {
    hid_t obj = H5Oopen(file_id_, path.c_str(), H5P_DEFAULT);
    if (obj < 0) return 0.0;
    double val = 0.0;
    if (H5Aexists(obj, name.c_str()) > 0) {
        hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
        H5Aread(attr, H5T_NATIVE_DOUBLE, &val);
        H5Aclose(attr);
    }
    H5Oclose(obj);
    return val;
}

std::string HDF5File::get_attr_string(const std::string& path, const std::string& name) {
    hid_t obj = H5Oopen(file_id_, path.c_str(), H5P_DEFAULT);
    if (obj < 0) return "";
    std::string val;
    if (H5Aexists(obj, name.c_str()) > 0) {
        hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
        hid_t type = H5Aget_type(attr);
        size_t sz = H5Tget_size(type);
        std::vector<char> buf(sz);
        H5Aread(attr, type, buf.data());
        val = std::string(buf.data());
        H5Tclose(type);
        H5Aclose(attr);
    }
    H5Oclose(obj);
    return val;
}

// --- 1D vector writers ---

static hid_t create_1d_dataset(hid_t file_id, const std::string& path,
                                hid_t dtype, hsize_t len, int compression) {
    // Delete if exists
    if (H5Lexists(file_id, path.c_str(), H5P_DEFAULT) > 0) {
        H5Ldelete(file_id, path.c_str(), H5P_DEFAULT);
    }

    // Create intermediate groups
    std::string group_path;
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        group_path = path.substr(0, pos);
        if (H5Lexists(file_id, group_path.c_str(), H5P_DEFAULT) <= 0) {
            hid_t g = H5Gcreate2(file_id, group_path.c_str(), H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(g);
        }
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

    if (compression > 0 && len > chunk_dims[0]) {
        H5Pclose(plist);
    }
    H5Sclose(space);
    return dset;
}

void HDF5File::write_vector_double(const std::string& path,
                                    const std::vector<double>& data, int compression) {
    hid_t dset = create_1d_dataset(file_id_, path, H5T_NATIVE_DOUBLE, data.size(), compression);
    herr_t status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    check_write(status, "H5Dwrite", path.c_str());
    H5Dclose(dset);
}

void HDF5File::write_vector_int32(const std::string& path, const std::vector<int32>& data) {
    hid_t dset = create_1d_dataset(file_id_, path, H5T_NATIVE_INT32, data.size(), 1);
    herr_t status = H5Dwrite(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    check_write(status, "H5Dwrite", path.c_str());
    H5Dclose(dset);
}

void HDF5File::write_vector_int64(const std::string& path, const std::vector<int64>& data) {
    hid_t dset = create_1d_dataset(file_id_, path, H5T_NATIVE_INT64, data.size(), 1);
    herr_t status = H5Dwrite(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    check_write(status, "H5Dwrite", path.c_str());
    H5Dclose(dset);
}

void HDF5File::write_vector_string(const std::string& path,
                                    const std::vector<std::string>& data) {
    // Create intermediate groups
    std::string group_path;
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        group_path = path.substr(0, pos);
        if (!group_path.empty() && H5Lexists(file_id_, group_path.c_str(), H5P_DEFAULT) <= 0) {
            hid_t g = H5Gcreate2(file_id_, group_path.c_str(), H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(g);
        }
    }

    // Variable-length strings
    hid_t strtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(strtype, H5T_VARIABLE);

    hsize_t len = data.size();
    std::vector<const char*> cstrs(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        cstrs[i] = data[i].c_str();
    }

    hsize_t chunk_dims[1] = {std::min(len, static_cast<hsize_t>(HDF5_CHUNK_SIZE_1D))};
    hid_t plist = H5P_DEFAULT;
    if (len > chunk_dims[0]) {
        plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 1, chunk_dims);
        H5Pset_deflate(plist, 1);
    }

    hid_t space = H5Screate_simple(1, &len, nullptr);
    hid_t dset = H5Dcreate2(file_id_, path.c_str(), strtype, space,
                             H5P_DEFAULT, plist, H5P_DEFAULT);
    herr_t status = H5Dwrite(dset, strtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, cstrs.data());
    check_write(status, "H5Dwrite", path.c_str());

    H5Dclose(dset);
    H5Sclose(space);
    if (len > chunk_dims[0]) H5Pclose(plist);
    H5Tclose(strtype);
}

// --- 1D vector readers ---

static hid_t open_dataset(hid_t file_id, const std::string& path) {
    if (H5Lexists(file_id, path.c_str(), H5P_DEFAULT) <= 0) {
        throw std::runtime_error("HDF5 dataset not found: " + path);
    }
    return H5Dopen2(file_id, path.c_str(), H5P_DEFAULT);
}

std::vector<double> HDF5File::read_vector_double(const std::string& path) {
    hid_t dset = open_dataset(file_id_, path);
    hid_t space = H5Dget_space(dset);
    hsize_t len;
    H5Sget_simple_extent_dims(space, &len, nullptr);
    std::vector<double> out(len);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    H5Sclose(space);
    H5Dclose(dset);
    return out;
}

std::vector<int32> HDF5File::read_vector_int32(const std::string& path) {
    hid_t dset = open_dataset(file_id_, path);
    hid_t space = H5Dget_space(dset);
    hsize_t len;
    H5Sget_simple_extent_dims(space, &len, nullptr);
    std::vector<int32> out(len);
    H5Dread(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    H5Sclose(space);
    H5Dclose(dset);
    return out;
}

std::vector<int64> HDF5File::read_vector_int64(const std::string& path) {
    hid_t dset = open_dataset(file_id_, path);
    hid_t space = H5Dget_space(dset);
    hsize_t len;
    H5Sget_simple_extent_dims(space, &len, nullptr);
    std::vector<int64> out(len);
    H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    H5Sclose(space);
    H5Dclose(dset);
    return out;
}

std::vector<std::string> HDF5File::read_vector_string(const std::string& path) {
    hid_t dset = open_dataset(file_id_, path);
    hid_t space = H5Dget_space(dset);
    hsize_t len;
    H5Sget_simple_extent_dims(space, &len, nullptr);

    hid_t strtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(strtype, H5T_VARIABLE);
    std::vector<char*> cbufs(len);
    H5Dread(dset, strtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, cbufs.data());

    std::vector<std::string> out(len);
    for (hsize_t i = 0; i < len; ++i) {
        if (cbufs[i]) {
            out[i] = std::string(cbufs[i]);
            std::free(cbufs[i]); // HDF5 allocates via malloc for variable-length
        }
    }

    H5Tclose(strtype);
    H5Sclose(space);
    H5Dclose(dset);
    return out;
}

// --- Matrix factory methods (stubs, implemented in respective files) ---

} // namespace sclean
