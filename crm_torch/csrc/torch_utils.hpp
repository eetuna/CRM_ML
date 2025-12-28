#pragma once

#include <torch/extension.h>
#include <Eigen/Dense>
#include <stdexcept>

namespace crm_torch {

/**
 * Zero-copy mapping utilities between torch::Tensor and Eigen types.
 *
 * These utilities provide efficient data access without unnecessary copies,
 * crucial for performance in the OpenMP parallel forward/backward passes.
 */

/**
 * Validate that a tensor is on CPU (required for Eigen mapping).
 */
inline void check_cpu(const torch::Tensor& tensor, const std::string& name) {
    if (!tensor.is_cpu()) {
        throw std::runtime_error(name + " must be on CPU. CUDA support not yet implemented.");
    }
}

/**
 * Validate tensor dtype is float64 (double).
 */
inline void check_float64(const torch::Tensor& tensor, const std::string& name) {
    if (tensor.dtype() != torch::kFloat64) {
        throw std::runtime_error(name + " must be float64 (double).");
    }
}

/**
 * Validate tensor is contiguous in memory (required for zero-copy mapping).
 */
inline void check_contiguous(const torch::Tensor& tensor, const std::string& name) {
    if (!tensor.is_contiguous()) {
        throw std::runtime_error(name + " must be contiguous.");
    }
}

/**
 * Full validation: CPU + float64 + contiguous.
 */
inline void validate_tensor(const torch::Tensor& tensor, const std::string& name) {
    check_cpu(tensor, name);
    check_float64(tensor, name);
    check_contiguous(tensor, name);
}

/**
 * Get batch size from a batched tensor.
 * Assumes tensor shape is (B, ...).
 */
inline int64_t get_batch_size(const torch::Tensor& tensor) {
    if (tensor.dim() < 1) {
        throw std::runtime_error("Tensor must have at least 1 dimension for batching.");
    }
    return tensor.size(0);
}

/**
 * Create Eigen::Map view of a 1D tensor section.
 *
 * @param tensor Input tensor of shape (B, N) or (N,)
 * @param batch_idx Batch index (ignored if tensor is 1D)
 * @param size Expected size N
 * @return Eigen::Map<Eigen::VectorXd> view (size N)
 */
inline Eigen::Map<Eigen::VectorXd> map_vector(torch::Tensor& tensor, int64_t batch_idx, int64_t size) {
    validate_tensor(tensor, "vector tensor");

    double* data_ptr = nullptr;
    if (tensor.dim() == 1) {
        // Unbatched tensor (size,)
        if (tensor.size(0) != size) {
            throw std::runtime_error("Expected vector size " + std::to_string(size) +
                                   " but got " + std::to_string(tensor.size(0)));
        }
        data_ptr = tensor.data_ptr<double>();
    } else if (tensor.dim() == 2) {
        // Batched tensor (B, size)
        if (tensor.size(1) != size) {
            throw std::runtime_error("Expected vector size " + std::to_string(size) +
                                   " but got " + std::to_string(tensor.size(1)));
        }
        data_ptr = tensor.data_ptr<double>() + batch_idx * size;
    } else {
        throw std::runtime_error("Vector tensor must be 1D or 2D, got " + std::to_string(tensor.dim()) + "D");
    }

    return Eigen::Map<Eigen::VectorXd>(data_ptr, size);
}

/**
 * Create Eigen::Map view of a 2D tensor section (per batch).
 *
 * @param tensor Input tensor of shape (B, rows, cols)
 * @param batch_idx Batch index
 * @param rows Expected number of rows
 * @param cols Expected number of columns
 * @return Eigen::Map<Eigen::MatrixXd> view (rows x cols, row-major)
 */
inline Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
map_matrix(torch::Tensor& tensor, int64_t batch_idx, int64_t rows, int64_t cols) {
    validate_tensor(tensor, "matrix tensor");

    if (tensor.dim() != 3) {
        throw std::runtime_error("Matrix tensor must be 3D (B, rows, cols), got " +
                               std::to_string(tensor.dim()) + "D");
    }
    if (tensor.size(1) != rows || tensor.size(2) != cols) {
        throw std::runtime_error("Expected matrix size (" + std::to_string(rows) + ", " +
                               std::to_string(cols) + ") but got (" +
                               std::to_string(tensor.size(1)) + ", " +
                               std::to_string(tensor.size(2)) + ")");
    }

    double* data_ptr = tensor.data_ptr<double>() + batch_idx * rows * cols;
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        data_ptr, rows, cols
    );
}

/**
 * Create const Eigen::Map view (read-only version of map_vector).
 */
inline Eigen::Map<const Eigen::VectorXd> map_vector_const(const torch::Tensor& tensor,
                                                           int64_t batch_idx,
                                                           int64_t size) {
    validate_tensor(tensor, "const vector tensor");

    const double* data_ptr = nullptr;
    if (tensor.dim() == 1) {
        if (tensor.size(0) != size) {
            throw std::runtime_error("Expected vector size " + std::to_string(size) +
                                   " but got " + std::to_string(tensor.size(0)));
        }
        data_ptr = tensor.data_ptr<double>();
    } else if (tensor.dim() == 2) {
        if (tensor.size(1) != size) {
            throw std::runtime_error("Expected vector size " + std::to_string(size) +
                                   " but got " + std::to_string(tensor.size(1)));
        }
        data_ptr = tensor.data_ptr<double>() + batch_idx * size;
    } else {
        throw std::runtime_error("Vector tensor must be 1D or 2D, got " + std::to_string(tensor.dim()) + "D");
    }

    return Eigen::Map<const Eigen::VectorXd>(data_ptr, size);
}

/**
 * Extract scalar value from a tensor (handle both scalar and batched tensors).
 *
 * @param tensor Input tensor (scalar or shape (B,))
 * @param batch_idx Batch index (used if tensor is batched)
 * @return double scalar value
 */
inline double get_scalar(const torch::Tensor& tensor, int64_t batch_idx) {
    check_cpu(tensor, "scalar tensor");
    check_float64(tensor, "scalar tensor");

    if (tensor.dim() == 0) {
        // Scalar tensor (single value for all batch elements)
        return tensor.item<double>();
    } else if (tensor.dim() == 1) {
        // Batched scalar (B,)
        if (tensor.size(0) == 1) {
            // Single value broadcasted to all batch elements
            return tensor[0].item<double>();
        } else {
            // Per-batch values
            return tensor[batch_idx].item<double>();
        }
    } else {
        throw std::runtime_error("Scalar tensor must be 0D or 1D, got " +
                               std::to_string(tensor.dim()) + "D");
    }
}

/**
 * Allocate output tensor with proper shape and dtype.
 *
 * @param batch_size Batch dimension
 * @param output_dim Output feature dimension
 * @return torch::Tensor of shape (B, output_dim), dtype float64, on CPU
 */
inline torch::Tensor allocate_output(int64_t batch_size, int64_t output_dim) {
    return torch::zeros({batch_size, output_dim},
                       torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
}

/**
 * Allocate Jacobian tensor (A or B matrix).
 *
 * @param batch_size Batch dimension
 * @param output_dim Output dimension (rows)
 * @param input_dim Input dimension (columns)
 * @return torch::Tensor of shape (B, output_dim, input_dim), dtype float64, on CPU
 */
inline torch::Tensor allocate_jacobian(int64_t batch_size, int64_t output_dim, int64_t input_dim) {
    return torch::zeros({batch_size, output_dim, input_dim},
                       torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
}

} // namespace crm_torch
