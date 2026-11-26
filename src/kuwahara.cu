#include "filters.h"

__device__ static void
region_stats(const uint8_t *in, int width, int height, int cx, int cy,
             int x0, int x1, int y0, int y1, float mean[3], float *stddev)
{
  float s[3]  = {0.0f, 0.0f, 0.0f};
  float ss[3] = {0.0f, 0.0f, 0.0f};
  int n = 0;

  for (int yy = y0; yy <= y1; ++yy)
  {
    int py = clampi(cy + yy, 0, height - 1);
    for (int xx = x0; xx <= x1; ++xx)
    {
      int px = clampi(cx + xx, 0, width - 1);
      int pi = 3 * (py * width + px);
      for (int c = 0; c < 3; ++c)
      {
        float v = (float)in[pi + c];
        s[c]  += v;
        ss[c] += v * v;
      }
      ++n;
    }
  }

  float inv = 1.0f / (float)n;
  float var_sum = 0.0f;
  for (int c = 0; c < 3; ++c)
  {
    mean[c] = s[c] * inv;
    float var = ss[c] * inv - mean[c] * mean[c];
    var_sum += (var > 0.0f) ? var : 0.0f;
  }
  *stddev = sqrtf(var_sum / 3.0f);
}

__global__ void
kuwahara_kernel(const uint8_t *in, uint8_t *out, int width, int height, int radius, float sharpness)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int r = radius;

  int qx0[4] = {-r,  0, -r,  0};
  int qx1[4] = { 0,  r,  0,  r};
  int qy0[4] = {-r, -r,  0,  0};
  int qy1[4] = { 0,  0,  r,  r};

  float means[4][3];
  float stds[4];

  float min_std = 1e30f;
  for (int q = 0; q < 4; ++q)
  {
    region_stats(in, width, height, x, y,
        qx0[q], qx1[q], qy0[q], qy1[q],
        means[q], &stds[q]);
    if (stds[q] < min_std) min_std = stds[q];
  }

  const float eps = 0.01f;
  float wsum = 0.0f;
  float acc[3] = {0.0f, 0.0f, 0.0f};

  for (int q = 0; q < 4; ++q)
  {
    float ratio = (stds[q] + eps) / (min_std + eps);
    float w = powf(1.0f / ratio, sharpness);
    wsum += w;
    for (int c = 0; c < 3; ++c)
      acc[c] += w * means[q][c];
  }

  int i = 3 * (y * width + x);
  float inv = 1.0f / wsum;
  for (int c = 0; c < 3; ++c)
  {
    float v = acc[c] * inv + 0.5f;
    out[i + c] = (uint8_t)fminf(fmaxf(v, 0.0f), 255.0f);
  }
}

void
filter_run_kuwahara(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
  dim3 block(16, 16);
  kuwahara_kernel<<<grid_for(w, h, block), block>>>(d_in, d_out, w, h, p->radius, p->sharpness);
  cudaDeviceSynchronize();
}
