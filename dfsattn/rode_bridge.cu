#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "RoDeSddmm.h"
#include "RoDeSpmm.h"

namespace {

struct SegmentData {
  std::vector<int> block_rows;
  std::vector<int> residue_rows;
  std::vector<int> segment_offsets;
};

// This is the host-side row partitioning contract consumed by the downloaded
// RoDe kernels.  The compute kernels themselves are compiled directly from
// RoDe_SDDMM/RoDe_SpMM; this only turns an exact CSR row pointer into RoDe's
// regular/residue descriptors.
SegmentData divide_rows(const std::vector<int>& row_offsets,
                        int segment_length) {
  constexpr int kVectorLength = 4;
  constexpr int kBlockLength = 32;
  SegmentData out;
  const int rows = static_cast<int>(row_offsets.size()) - 1;
  out.block_rows.reserve(rows);
  out.residue_rows.reserve(rows);
  out.segment_offsets.reserve(rows + 1);

  for (int row = 0; row < rows; ++row) {
    int row_offset = row_offsets[row];
    const int padding = row_offset % kVectorLength;
    int nnz = row_offsets[row + 1] - row_offset + padding;

    if (nnz > segment_length) {
      out.block_rows.push_back(row);
      out.segment_offsets.push_back(row_offset);
      row_offset = (row_offset + segment_length) - padding;
      nnz -= segment_length;
    }
    while (nnz > segment_length) {
      out.block_rows.push_back(row);
      out.segment_offsets.push_back(row_offset);
      row_offset += segment_length;
      nnz -= segment_length;
    }
    if (nnz > 0) {
      if (nnz >= kBlockLength) {
        out.block_rows.push_back(row);
        out.segment_offsets.push_back(row_offset);
      }
      if (nnz % kBlockLength) {
        out.residue_rows.push_back(row);
      }
    }
  }
  out.segment_offsets.push_back(row_offsets.back());
  return out;
}

torch::Tensor to_cuda_int(const std::vector<int>& values) {
  auto cpu_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  auto gpu_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto cpu = torch::from_blob(
      const_cast<int*>(values.data()),
      {static_cast<long>(values.size())}, cpu_options).clone();
  return cpu.to(gpu_options);
}

class RoDeCenterPlan {
 public:
  RoDeCenterPlan(torch::Tensor row_offsets_cpu,
                 torch::Tensor column_indices_cpu,
                 int columns,
                 int sddmm_segment_length,
                 int spmm_segment_length)
      : rows_(static_cast<int>(row_offsets_cpu.numel()) - 1),
        columns_(columns),
        nnz_(static_cast<int>(column_indices_cpu.numel())) {
    TORCH_CHECK(row_offsets_cpu.device().is_cpu(),
                "RoDe row_offsets must be a CPU tensor");
    TORCH_CHECK(column_indices_cpu.device().is_cpu(),
                "RoDe column_indices must be a CPU tensor");
    TORCH_CHECK(row_offsets_cpu.scalar_type() == torch::kInt32,
                "RoDe row_offsets must be int32");
    TORCH_CHECK(column_indices_cpu.scalar_type() == torch::kInt32,
                "RoDe column_indices must be int32");
    TORCH_CHECK(row_offsets_cpu.dim() == 1 && column_indices_cpu.dim() == 1,
                "RoDe CSR inputs must be 1D");
    TORCH_CHECK(row_offsets_cpu.is_contiguous() && column_indices_cpu.is_contiguous(),
                "RoDe CSR inputs must be contiguous");
    TORCH_CHECK(row_offsets_cpu.numel() >= 1 &&
                    row_offsets_cpu[-1].item<int>() == nnz_,
                "RoDe CSR row_offsets does not match column_indices");

    const int* row_ptr = row_offsets_cpu.data_ptr<int>();
    const int* col_ptr = column_indices_cpu.data_ptr<int>();
    std::vector<int> row_offsets(row_ptr, row_ptr + rows_ + 1);
    std::vector<int> column_values(col_ptr, col_ptr + nnz_);
    TORCH_CHECK(columns_ >= 0, "RoDe columns must be non-negative");
    TORCH_CHECK(!column_values.empty() || rows_ == 0,
                "RoDe requires a non-empty CSR for this probe");
    if (!column_values.empty()) {
      TORCH_CHECK(*std::max_element(column_values.begin(), column_values.end()) < columns_,
                  "RoDe CSR column index exceeds dense matrix columns");
    }

    auto device = torch::Device(torch::kCUDA);
    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    row_offsets_ = torch::empty({rows_ + 1}, int_options);
    column_indices_ = torch::empty({nnz_}, int_options);
    row_offsets_.copy_(row_offsets_cpu);
    column_indices_.copy_(column_indices_cpu);
    values_ones_ = torch::ones({nnz_}, float_options);

    // RoDe's published SDDMM and SpMM paths use different regular segment
    // lengths, as in the original evaluation programs.
    auto sddmm = divide_rows(row_offsets, sddmm_segment_length);
    auto spmm = divide_rows(row_offsets, spmm_segment_length);
    sddmm_block_rows_ = to_cuda_int(sddmm.block_rows);
    sddmm_residue_rows_ = to_cuda_int(sddmm.residue_rows);
    sddmm_segment_offsets_ = to_cuda_int(sddmm.segment_offsets);
    spmm_block_rows_ = to_cuda_int(spmm.block_rows);
    spmm_residue_rows_ = to_cuda_int(spmm.residue_rows);
    spmm_segment_offsets_ = to_cuda_int(spmm.segment_offsets);
    sddmm_block_count_ = static_cast<int>(sddmm.block_rows.size());
    sddmm_residue_count_ = static_cast<int>(sddmm.residue_rows.size());
    spmm_block_count_ = static_cast<int>(spmm.block_rows.size());
    spmm_residue_count_ = static_cast<int>(spmm.residue_rows.size());
  }

  torch::Tensor sddmm(torch::Tensor lhs, torch::Tensor rhs) const {
    check_dense(lhs, "lhs");
    check_dense(rhs, "rhs");
    TORCH_CHECK(lhs.size(0) == rows_, "lhs row count does not match CSR");
    TORCH_CHECK(rhs.size(0) == columns_, "rhs row count does not match CSR");
    TORCH_CHECK(lhs.size(1) == 128 && rhs.size(1) == 128,
                "The downloaded RoDe n128 path requires head_dim=128");
    auto out = torch::empty({nnz_}, lhs.options());
    auto stream = at::cuda::getCurrentCUDAStream(lhs.device().index()).stream();
    RoDeSDDMM_n128(
        sddmm_block_count_, sddmm_residue_count_, columns_, 128,
        ptr(sddmm_block_rows_), ptr(sddmm_residue_rows_), ptr(sddmm_segment_offsets_),
        ptr(row_offsets_), ptr(column_indices_), float_ptr(values_ones_),
        lhs.data_ptr<float>(), rhs.data_ptr<float>(), out.data_ptr<float>(),
        stream, stream);
    return out;
  }

  torch::Tensor spmm(torch::Tensor values, torch::Tensor dense) const {
    TORCH_CHECK(values.is_cuda() && dense.is_cuda(),
                "RoDe SpMM inputs must be CUDA tensors");
    TORCH_CHECK(values.scalar_type() == torch::kFloat32 &&
                    dense.scalar_type() == torch::kFloat32,
                "RoDe SpMM inputs must be float32");
    TORCH_CHECK(values.dim() == 1 && values.numel() == nnz_,
                "RoDe values must have one entry per CSR nonzero");
    TORCH_CHECK(dense.dim() == 2 && dense.size(0) == columns_ &&
                    dense.size(1) == 128,
                "RoDe n128 SpMM expects [columns,128] dense input");
    auto out = torch::zeros({rows_, 128}, dense.options());
    auto stream = at::cuda::getCurrentCUDAStream(dense.device().index()).stream();
    RoDeSpmm_n128(
        spmm_block_count_, spmm_residue_count_, columns_, 128,
        values.data_ptr<float>(), ptr(column_indices_), ptr(row_offsets_),
        ptr(spmm_block_rows_), ptr(spmm_residue_rows_), ptr(spmm_segment_offsets_),
        dense.data_ptr<float>(), out.data_ptr<float>(), stream, stream);
    return out;
  }

  int rows() const { return rows_; }
  int columns() const { return columns_; }
  int nnz() const { return nnz_; }
  int sddmm_block_count() const { return sddmm_block_count_; }
  int sddmm_residue_count() const { return sddmm_residue_count_; }
  int spmm_block_count() const { return spmm_block_count_; }
  int spmm_residue_count() const { return spmm_residue_count_; }

 private:
  static void check_dense(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                name, " must be float32 for the downloaded RoDe kernel");
    TORCH_CHECK(tensor.dim() == 2 && tensor.is_contiguous(),
                name, " must be contiguous 2D");
  }

  static int* ptr(const torch::Tensor& tensor) {
    return tensor.numel() == 0 ? nullptr : tensor.data_ptr<int>();
  }

  static float* float_ptr(const torch::Tensor& tensor) {
    return tensor.numel() == 0 ? nullptr : tensor.data_ptr<float>();
  }

  int rows_;
  int columns_;
  int nnz_;
  int sddmm_block_count_;
  int sddmm_residue_count_;
  int spmm_block_count_;
  int spmm_residue_count_;
  torch::Tensor row_offsets_;
  torch::Tensor column_indices_;
  torch::Tensor values_ones_;
  torch::Tensor sddmm_block_rows_;
  torch::Tensor sddmm_residue_rows_;
  torch::Tensor sddmm_segment_offsets_;
  torch::Tensor spmm_block_rows_;
  torch::Tensor spmm_residue_rows_;
  torch::Tensor spmm_segment_offsets_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<RoDeCenterPlan>(m, "RoDeCenterPlan")
      .def(pybind11::init<torch::Tensor, torch::Tensor, int, int, int>(),
           pybind11::arg("row_offsets_cpu"),
           pybind11::arg("column_indices_cpu"),
           pybind11::arg("columns"),
           pybind11::arg("sddmm_segment_length") = 32,
           pybind11::arg("spmm_segment_length") = 512)
      .def("sddmm", &RoDeCenterPlan::sddmm)
      .def("spmm", &RoDeCenterPlan::spmm)
      .def_property_readonly("rows", &RoDeCenterPlan::rows)
      .def_property_readonly("columns", &RoDeCenterPlan::columns)
      .def_property_readonly("nnz", &RoDeCenterPlan::nnz)
      .def_property_readonly("sddmm_block_count", &RoDeCenterPlan::sddmm_block_count)
      .def_property_readonly("sddmm_residue_count", &RoDeCenterPlan::sddmm_residue_count)
      .def_property_readonly("spmm_block_count", &RoDeCenterPlan::spmm_block_count)
      .def_property_readonly("spmm_residue_count", &RoDeCenterPlan::spmm_residue_count);
}
