#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

cudaError_t attention_k_scale_off_wide_launch(
    const ninfer::Tensor& q, const ninfer::Tensor& positions,
    const ninfer::Tensor& table_rows, float scale,
    const ninfer::PagedKVBatchLayerView& cache, ninfer::Tensor& out, cudaStream_t stream);

cudaError_t attention_k_scale_on_wide_launch(
    const ninfer::Tensor& q, const ninfer::Tensor& positions,
    const ninfer::Tensor& table_rows, float scale,
    const ninfer::PagedKVBatchLayerView& cache, ninfer::Tensor& out, cudaStream_t stream);
