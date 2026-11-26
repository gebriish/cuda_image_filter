#include "filters.h"

__global__ void
histogram_kernel(const uint8_t *in, uint32_t *d_hist, int width, int height, int total_pixels)
{
  __shared__ uint32_t hist[256];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int threads_per_block = blockDim.x * blockDim.y;

  for (int c = tid; c < 256; c += threads_per_block)
    hist[c] = 0;
  __syncthreads();

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height)
  {
    int i = 3 * (y * width + x);
    uint8_t r = in[i + 0];
    uint8_t g = in[i + 1];
    uint8_t b = in[i + 2];

    float z = (0.299f * r + 0.587f * g + 0.114f * b + 0.5f);
    uint32_t val = (z > 255.0f) ? 255u : (uint32_t)z;

    atomicAdd(&hist[val], 1u);
  }
  __syncthreads();

  for (int c = tid; c < 256; c += threads_per_block)
    atomicAdd(&d_hist[c], hist[c]);
}

__global__ void
cdf_kernel(const uint32_t *d_hist, uint8_t *d_fval, int width, int height, int total_pixels)
{
  __shared__ uint32_t cdf[256];
  __shared__ uint32_t warps_total[8];
  __shared__ uint32_t cdf_min;

  if (threadIdx.x >= 256) return;
  int tid = threadIdx.x;

  int lane    = tid & 31;
  int warp_id = tid >> 5;

  uint32_t sum = d_hist[tid];
  for (int i = 1; i < 32; i *= 2)
  {
    uint32_t x = __shfl_up_sync(0xffffffff, sum, i);
    if (lane >= i) sum += x;
  }
  __syncthreads();

  if (lane == 31) warps_total[warp_id] = sum;
  __syncthreads();

  if (warp_id == 0)
  {
    uint32_t warp_sum = 0;
    if (lane < 8) warp_sum = warps_total[lane];

    for (int i = 1; i < 8; i *= 2)
    {
      uint32_t x = __shfl_up_sync(0xffffffff, warp_sum, i);
      if (lane >= i && lane < 8) warp_sum += x;
    }

    if (lane < 8) warps_total[lane] = warp_sum;
  }
  __syncthreads();

  if (warp_id > 0) sum += warps_total[warp_id - 1];

  cdf[tid] = sum;
  __syncthreads();

  if (tid == 0)
  {
    for (int i = 0; i < 256; i++)
    {
      if (cdf[i] > 0)
      {
        cdf_min = cdf[i];
        break;
      }
    }
  }
  __syncthreads();

  float num = sum - cdf_min;
  float den = total_pixels - cdf_min;
  float val = 0;
  if (den != 0) val = (num / den) * 255.0f + 0.5f;
  d_fval[tid] = (uint8_t)val;
}

__global__ void
writeback_histogram_kernel(const uint8_t *in, uint8_t *out, const uint8_t *d_fval, int w, int h, int total_pixels)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  int idx = 3 * (y * w + x);
  uint8_t r = in[idx + 0];
  uint8_t g = in[idx + 1];
  uint8_t b = in[idx + 2];

  float z = (0.299f * r + 0.587f * g + 0.114f * b + 0.5f);
  uint32_t val = (z > 255.0f) ? 255u : (uint32_t)z;

  float new_val = d_fval[val];
  float scale = 1;
  if (z != 0.5f) scale = new_val / (z - 0.5f);

  for (int i = 0; i < 3; i++)
  {
    float v;
    uint8_t update;
    if (i == 0) v = r * scale;
    if (i == 1) v = g * scale;
    if (i == 2) v = b * scale;

    v += 0.5f;
    if (v <= 0.0f)        update = 0;
    else if (v >= 255.0f) update = 255;
    else                  update = (uint8_t)v;
    out[idx + i] = update;
  }
}

void
filter_run_histogram(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
  dim3 block(16, 16);
  int total_pixels = w * h;

  uint32_t *d_hist = 0;
  uint8_t  *d_final_vals = 0;
  cudaMalloc(&d_hist, 256 * sizeof(uint32_t));
  cudaMalloc(&d_final_vals, 256 * sizeof(uint8_t));
  cudaMemset(d_hist, 0, 256 * sizeof(uint32_t));

  histogram_kernel<<<grid_for(w, h, block), block>>>(d_in, d_hist, w, h, total_pixels);
  cdf_kernel<<<1, 256>>>(d_hist, d_final_vals, w, h, total_pixels);
  writeback_histogram_kernel<<<grid_for(w, h, block), block>>>(d_in, d_out, d_final_vals, w, h, total_pixels);
  cudaDeviceSynchronize();

  cudaFree(d_hist);
  cudaFree(d_final_vals);
}
