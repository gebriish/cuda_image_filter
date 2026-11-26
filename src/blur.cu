#include "filters.h"

__constant__ float c_weights[2 * MAX_RADIUS + 1];

__global__ void
blur_h_kernel(const uint8_t *in, float *tmp, int width, int height, int radius)
{
  extern __shared__ uint8_t s_tile[];
  const int tile_w = BLOCK_X + 2 * radius;

  int x  = blockIdx.x * BLOCK_X + threadIdx.x;
  int y  = blockIdx.y * BLOCK_Y + threadIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int gy = clampi(y, 0, height - 1);
  for (int sx = tx; sx < tile_w; sx += BLOCK_X)
  {
    int gx = clampi(blockIdx.x * BLOCK_X + sx - radius, 0, width - 1);
    int gi = 3 * (gy * width + gx);
    int si = 3 * (ty * tile_w + sx);
    s_tile[si + 0] = in[gi + 0];
    s_tile[si + 1] = in[gi + 1];
    s_tile[si + 2] = in[gi + 2];
  }

  __syncthreads();

  if (x >= width || y >= height) return;

  float acc[3] = {0.0f, 0.0f, 0.0f};
  for (int k = -radius; k <= radius; ++k)
  {
    float w = c_weights[k + radius];
    int si = 3 * (ty * tile_w + (tx + radius + k));
    acc[0] += w * s_tile[si + 0];
    acc[1] += w * s_tile[si + 1];
    acc[2] += w * s_tile[si + 2];
  }

  int i = 3 * (y * width + x);
  tmp[i + 0] = acc[0];
  tmp[i + 1] = acc[1];
  tmp[i + 2] = acc[2];
}

__global__ void
blur_v_kernel(const float *tmp, uint8_t *out, int width, int height, int radius)
{
  int x = blockIdx.x * BLOCK_X + threadIdx.x;
  int y = blockIdx.y * BLOCK_Y + threadIdx.y;
  if (x >= width || y >= height) return;

  float acc[3] = {0.0f, 0.0f, 0.0f};
  for (int k = -radius; k <= radius; ++k)
  {
    float w = c_weights[k + radius];
    int sy = clampi(y + k, 0, height - 1);
    int si = 3 * (sy * width + x);
    acc[0] += w * tmp[si + 0];
    acc[1] += w * tmp[si + 1];
    acc[2] += w * tmp[si + 2];
  }

  int i = 3 * (y * width + x);
  for (int c = 0; c < 3; ++c)
  {
    float v = acc[c] + 0.5f;
    out[i + c] = (uint8_t)fminf(fmaxf(v, 0.0f), 255.0f);
  }
}

static void
blur_upload_weights(int radius, float sigma)
{
  float h_w[2 * MAX_RADIUS + 1];
  float sum = 0.0f;

  for (int k = -radius; k <= radius; ++k)
  {
    float w = expf(-(float)(k * k) / (2.0f * sigma * sigma));
    h_w[k + radius] = w;
    sum += w;
  }
  for (int k = 0; k <= 2 * radius; ++k)
    h_w[k] /= sum;

  cudaMemcpyToSymbol(c_weights, h_w, (2 * radius + 1) * sizeof(float));
}

void
filter_run_blur(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
  int radius  = p->radius;
  float sigma = p->sigma;
  if (sigma <= 0.0f) sigma = radius / 3.0f;

  blur_upload_weights(radius, sigma);

  float *d_tmp = 0;
  cudaMalloc(&d_tmp, (size_t)w * h * 3 * sizeof(float));

  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid = grid_for(w, h, block);
  size_t h_smem = (size_t)(BLOCK_X + 2 * radius) * BLOCK_Y * 3 * sizeof(uint8_t);

  blur_h_kernel<<<grid, block, h_smem>>>(d_in, d_tmp, w, h, radius);
  blur_v_kernel<<<grid, block>>>(d_tmp, d_out, w, h, radius);
  cudaDeviceSynchronize();

  cudaFree(d_tmp);
}
