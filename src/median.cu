#include "filters.h"
#include <stdio.h>

__global__ void
median_kernel(const uint8_t * __restrict__ in,
              uint8_t * __restrict__ out,
              int width, int height, int radius)
{
  extern __shared__ uint8_t s_tile[];
  const int tile_w = BLOCK_X + 2 * radius;
  const int tile_h = BLOCK_Y + 2 * radius;

  int x  = blockIdx.x * BLOCK_X + threadIdx.x;
  int y  = blockIdx.y * BLOCK_Y + threadIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Cooperatively load 2D tile + halo into shared memory
  for (int sy = ty; sy < tile_h; sy += BLOCK_Y)
  {
    int gy = clampi(blockIdx.y * BLOCK_Y + sy - radius, 0, height - 1);
    for (int sx = tx; sx < tile_w; sx += BLOCK_X)
    {
      int gx = clampi(blockIdx.x * BLOCK_X + sx - radius, 0, width - 1);
      int gi = 3 * (gy * width + gx);
      int si = 3 * (sy * tile_w + sx);
      s_tile[si + 0] = in[gi + 0];
      s_tile[si + 1] = in[gi + 1];
      s_tile[si + 2] = in[gi + 2];
    }
  }

  __syncthreads();

  if (x >= width || y >= height) return;

  const int window_area = (2 * radius + 1) * (2 * radius + 1);
  const int median_pos  = window_area / 2;
  int out_idx = 3 * (y * width + x);

  // Compute exact median per RGB channel via 256-bin histogram
  for (int c = 0; c < 3; ++c)
  {
    uint16_t hist[256] = {0};

    for (int dy = -radius; dy <= radius; ++dy)
    {
      int row_offset = (ty + radius + dy) * tile_w;
      for (int dx = -radius; dx <= radius; ++dx)
      {
        int si = 3 * (row_offset + (tx + radius + dx)) + c;
        hist[s_tile[si]]++;
      }
    }

    int count = 0;
    uint8_t median_val = 0;
    for (int v = 0; v < 256; ++v)
    {
      count += hist[v];
      if (count > median_pos)
      {
        median_val = (uint8_t)v;
        break;
      }
    }

    out[out_idx + c] = median_val;
  }
}

void
filter_run_median(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
  int radius = clampi(p->radius, 0, MAX_RADIUS);
  if (radius == 0)
  {
    cudaMemcpy(d_out, d_in, (size_t)w * h * 3 * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
    return;
  }

  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid = grid_for(w, h, block);
  size_t smem = (size_t)(BLOCK_X + 2 * radius) * (BLOCK_Y + 2 * radius) * 3 * sizeof(uint8_t);

  // Check against standard 48 KB shared memory limit
  if (smem > 49152)
  {
    fprintf(stderr, "median: shared memory (%zu bytes) exceeds 48 KB limit; reduce radius or block size\n", smem);
    return;
  }

  median_kernel<<<grid, block, smem>>>(d_in, d_out, w, h, radius);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    fprintf(stderr, "median_kernel launch failed: %s\n", cudaGetErrorString(err));

  cudaDeviceSynchronize();
}
