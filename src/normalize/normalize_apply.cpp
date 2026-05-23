#include "normalize_internal.h"
#include "normalize_operator.h"       // for NormalizeMethod values
#include "hdf5/hdf5_sparse_writer.h"
#include <cmath>
#include <algorithm>

namespace sclean {

void normalize_sparse_chunk(NormalizeMethod method,
    const double* in_data, const int32* in_indices,
    const int64* in_indptr, int64 col_count,
    const double* size_factors, double scale_factor, bool do_pseudocount,
    HDF5SparseWriter* writer) {

    switch (method) {
        case NormalizeMethod::LogNormalize:
            for (int64 j = 0; j < col_count; ++j) {
                double sf = size_factors[j];
                double norm_factor = (sf > 0) ? scale_factor / sf : 1.0;
                int64 start = in_indptr[j] - in_indptr[0];
                int64 end = in_indptr[j + 1] - in_indptr[0];
                int64 nnz = end - start;

                std::vector<double> out_vals(nnz);
                for (int64 k = 0; k < nnz; ++k) {
                    out_vals[k] = std::log1p(in_data[start + k] * norm_factor);
                }
                writer->write_column(out_vals.data(), in_indices + start, nnz);
            }
            break;

        case NormalizeMethod::RelativeCounts:
            for (int64 j = 0; j < col_count; ++j) {
                double sf = size_factors[j];
                double norm_factor = (sf > 0) ? scale_factor / sf : 1.0;
                int64 start = in_indptr[j] - in_indptr[0];
                int64 end = in_indptr[j + 1] - in_indptr[0];
                int64 nnz = end - start;

                std::vector<double> out_vals(nnz);
                for (int64 k = 0; k < nnz; ++k) {
                    out_vals[k] = in_data[start + k] * norm_factor;
                }
                writer->write_column(out_vals.data(), in_indices + start, nnz);
            }
            break;

        case NormalizeMethod::CLR:
            // CLR does not preserve zeros — write each non-zero result entry
            for (int64 j = 0; j < col_count; ++j) {
                double gm = size_factors[j];
                int64 start = in_indptr[j] - in_indptr[0];
                int64 end = in_indptr[j + 1] - in_indptr[0];
                int64 nnz = end - start;

                std::vector<double> col_vals;
                std::vector<int32> col_idx;
                for (int64 k = 0; k < nnz; ++k) {
                    double v = in_data[start + k];
                    double pseudo = do_pseudocount ? 1.0 : 0.0;
                    double r = std::log((v + pseudo) / gm);
                    if (r != 0.0) {
                        col_vals.push_back(r);
                        col_idx.push_back(in_indices[start + k]);
                    }
                }
                writer->write_column(col_vals.data(), col_idx.data(),
                                     static_cast<int64>(col_vals.size()));
            }
            break;
    }
}

void normalize_dense_chunk(NormalizeMethod method,
    const double* in_buf, int64 n_rows, int64 n_cols,
    const double* size_factors, double scale_factor, bool do_pseudocount,
    HDF5SparseWriter* writer) {

    for (int64 j = 0; j < n_cols; ++j) {
        double sf = size_factors[j];
        double norm_factor = (sf > 0) ? scale_factor / sf : 1.0;

        std::vector<double> col_vals;
        std::vector<int32> col_idx;

        for (int64 i = 0; i < n_rows; ++i) {
            double v = in_buf[i * n_cols + j];
            double r = 0.0;

            switch (method) {
                case NormalizeMethod::LogNormalize:
                    r = std::log1p(v * norm_factor);
                    break;
                case NormalizeMethod::CLR: {
                    double pseudo = do_pseudocount ? 1.0 : 0.0;
                    r = std::log((v + pseudo) / sf);
                    break;
                }
                case NormalizeMethod::RelativeCounts:
                    r = v * norm_factor;
                    break;
            }

            if (r != 0.0) {
                col_vals.push_back(r);
                col_idx.push_back(static_cast<int32>(i));
            }
        }

        writer->write_column(col_vals.data(), col_idx.data(),
                             static_cast<int64>(col_vals.size()));
    }
}

} // namespace sclean
