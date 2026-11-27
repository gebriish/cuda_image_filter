#include "filters.h"
#include <math.h>
#include <stdio.h>

__constant__ float c_canny_weights[2 * MAX_RADIUS + 1];

// Stage 1a: Grayscale + Horizontal Gaussian Blur
__global__ void
canny_blur_h_kernel(const uint8_t * __restrict__ in,
                    float * __restrict__ tmp,
                    int width, int height, int radius)
{
  extern __shared__ float s_tile_f[];
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
    // Rec. 601 Luma conversion
    float gray = 0.299f * in[gi + 0] + 0.587f * in[gi + 1] + 0.114f * in[gi + 2];
    s_tile_f[ty * tile_w + sx] = gray;
  }

  __syncthreads();

  if (x >= width || y >= height) return;

  float acc = 0.0f;
  for (int k = -radius; k <= radius; ++k)
    acc += c_canny_weights[k + radius] * s_tile_f[ty * tile_w + (tx + radius + k)];

  tmp[y * width + x] = acc;
}

// Stage 1b: Vertical Gaussian Blur
__global__ void
canny_blur_v_kernel(const float * __restrict__ tmp,
                    float * __restrict__ smooth,
                    int width, int height, int radius)
{
  int x = blockIdx.x * BLOCK_X + threadIdx.x;
  int y = blockIdx.y * BLOCK_Y + threadIdx.y;
  if (x >= width || y >= height) return;

  float acc = 0.0f;
  for (int k = -radius; k <= radius; ++k)
  {
    int sy = clampi(y + k, 0, height - 1);
    acc += c_canny_weights[k + radius] * tmp[sy * width + x];
  }
  smooth[y * width + x] = acc;
}

// Stage 2: Sobel Gradient Magnitude & Quantized Direction (0=0°, 1=45°, 2=90°, 3=135°)
__global__ void
canny_sobel_kernel(const float * __restrict__ smooth,
                   float * __restrict__ mag,
                   uint8_t * __restrict__ dir,
                   int width, int height)
{
  extern __shared__ float s_sobel[];
  const int tile_w = BLOCK_X + 2;
  const int tile_h = BLOCK_Y + 2;

  int x  = blockIdx.x * BLOCK_X + threadIdx.x;
  int y  = blockIdx.y * BLOCK_Y + threadIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  for (int sy = ty; sy < tile_h; sy += BLOCK_Y)
  {
    int gy = clampi(blockIdx.y * BLOCK_Y + sy - 1, 0, height - 1);
    for (int sx = tx; sx < tile_w; sx += BLOCK_X)
    {
      int gx = clampi(blockIdx.x * BLOCK_X + sx - 1, 0, width - 1);
      s_sobel[sy * tile_w + sx] = smooth[gy * width + gx];
    }
  }

  __syncthreads();

  if (x >= width || y >= height) return;

  int r0 = (ty + 0) * tile_w + (tx + 1);
  int r1 = (ty + 1) * tile_w + (tx + 1);
  int r2 = (ty + 2) * tile_w + (tx + 1);

  float gx = -s_sobel[r0 - 1] + s_sobel[r0 + 1]
             -2.0f * s_sobel[r1 - 1] + 2.0f * s_sobel[r1 + 1]
             -s_sobel[r2 - 1] + s_sobel[r2 + 1];

  float gy = -s_sobel[r0 - 1] - 2.0f * s_sobel[r0] - s_sobel[r0 + 1]
             +s_sobel[r2 - 1] + 2.0f * s_sobel[r2] + s_sobel[r2 + 1];

  int idx = y * width + x;
  mag[idx] = hypotf(gx, gy);

  // Quantize gradient angle into 4 bins
  float angle = atan2f(gy, gx) * (180.0f / 3.14159265f);
  if (angle < 0.0f) angle += 180.0f;

  if ((angle < 22.5f) || (angle >= 157.5f))
    dir[idx] = 0; // Horizontal edge (0°) -> compare left/right
  else if (angle < 67.5f)
    dir[idx] = 1; // Diagonal (45°) -> compare top-right/bottom-left
  else if (angle < 112.5f)
    dir[idx] = 2; // Vertical edge (90°) -> compare top/bottom
  else
    dir[idx] = 3; // Diagonal (135°) -> compare top-left/bottom-right
}

// Stage 3: Non-Maximum Suppression (1-pixel edge thinning)
__global__ void
canny_nms_kernel(const float * __restrict__ mag,
                 const uint8_t * __restrict__ dir,
                 float * __restrict__ nms,
                 int width, int height)
{
  int x = blockIdx.x * BLOCK_X + threadIdx.x;
  int y = blockIdx.y * BLOCK_Y + threadIdx.y;
  if (x >= width || y >= height) return;

  int idx = y * width + x;
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
  {
    nms[idx] = 0.0f;
    return;
  }

  float m = mag[idx];
  float m1 = 0.0f, m2 = 0.0f;

  switch (dir[idx])
  {
    case 0: // 0°
      m1 = mag[y * width + (x - 1)];
      m2 = mag[y * width + (x + 1)];
      break;
    case 1: // 45°
      m1 = mag[(y - 1) * width + (x + 1)];
      m2 = mag[(y + 1) * width + (x - 1)];
      break;
    case 2: // 90°
      m1 = mag[(y - 1) * width + x];
      m2 = mag[(y + 1) * width + x];
      break;
    case 3: // 135°
      m1 = mag[(y - 1) * width + (x - 1)];
      m2 = mag[(y + 1) * width + (x + 1)];
      break;
  }

  nms[idx] = (m >= m1 && m >= m2) ? m : 0.0f;
}

// Stage 4: Double Threshold + 8-Neighbor Hysteresis -> RGB Output
__global__ void
canny_hysteresis_kernel(const float * __restrict__ nms,
                        uint8_t * __restrict__ out,
                        int width, int height,
                        float low_thresh, float high_thresh)
{
  int x = blockIdx.x * BLOCK_X + threadIdx.x;
  int y = blockIdx.y * BLOCK_Y + threadIdx.y;
  if (x >= width || y >= height) return;

  int idx = y * width + x;
  float v = nms[idx];
  uint8_t edge = 0;

  if (v >= high_thresh)
  {
    edge = 255;
  }
  else if (v >= low_thresh && x > 0 && x < width - 1 && y > 0 && y < height - 1)
  {
    // Promote weak edge if any 8-connected neighbor is a strong edge
    for (int dy = -1; dy <= 1; ++dy)
    {
      for (int dx = -1; dx <= 1; ++dx)
      {
        if (nms[(y + dy) * width + (x + dx)] >= high_thresh)
        {
          edge = 255;
          break;
        }
      }
      if (edge) break;
    }
  }

  int oi = 3 * idx;
  out[oi + 0] = edge;
  out[oi + 1] = edge;
  out[oi + 2] = edge;
}

static void
canny_upload_weights(int radius, float sigma)
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

  cudaMemcpyToSymbol(c_canny_weights, h_w, (2 * radius + 1) * sizeof(float));
}

void
filter_run_canny(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
  // Default pre-blur parameters if omitted
  int radius  = (p->radius > 0) ? clampi(p->radius, 1, MAX_RADIUS) : 2;
  float sigma = (p->sigma > 0.0f) ? p->sigma : 1.4f;

  // Scale thresholds (defaults: low=50, high=120 on 0..255 gradient scale)
  // If your Params struct has low_thresh / high_thresh fields, replace these two lines:
  float high_thresh = 120.0f;
  float low_thresh  = 50.0f;

  canny_upload_weights(radius, sigma);

  size_t num_pixels = (size_t)w * h;
  float *d_tmp = NULL, *d_smooth = NULL, *d_mag = NULL, *d_nms = NULL;
  uint8_t *d_dir = NULL;

  cudaMalloc(&d_tmp,    num_pixels * sizeof(float));
  cudaMalloc(&d_smooth, num_pixels * sizeof(float));
  cudaMalloc(&d_mag,    num_pixels * sizeof(float));
  cudaMalloc(&d_nms,    num_pixels * sizeof(float));
  cudaMalloc(&d_dir,    num_pixels * sizeof(uint8_t));

  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid = grid_for(w, h, block);

  size_t blur_smem  = (size_t)(BLOCK_X + 2 * radius) * BLOCK_Y * sizeof(float);
  size_t sobel_smem = (size_t)(BLOCK_X + 2) * (BLOCK_Y + 2) * sizeof(float);

  canny_blur_h_kernel<<<grid, block, blur_smem>>>(d_in, d_tmp, w, h, radius);
  canny_blur_v_kernel<<<grid, block>>>(d_tmp, d_smooth, w, h, radius);
  canny_sobel_kernel<<<grid, block, sobel_smem>>>(d_smooth, d_mag, d_dir, w, h);
  canny_nms_kernel<<<grid, block>>>(d_mag, d_dir, d_nms, w, h);
  canny_hysteresis_kernel<<<grid, block>>>(d_nms, d_out, w, h, low_thresh, high_thresh);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    fprintf(stderr, "canny kernel launch failed: %s\n", cudaGetErrorString(err));

  cudaDeviceSynchronize();

  cudaFree(d_tmp);
  cudaFree(d_smooth);
  cudaFree(d_mag);
  cudaFree(d_nms);
  cudaFree(d_dir);
}
