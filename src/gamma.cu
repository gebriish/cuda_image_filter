#include "filters.h"

__global__ void
gamma_kernel(const uint8_t *in, uint8_t *out, int width, int height, float gamma)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int i = 3 * (y * width + x);
  for (int j = 0; j < 3; j++)
  {
    float normal_val = in[i + j] / 255.0f;
    float correction = powf(normal_val, gamma);
    out[i + j] = (uint8_t)(correction * 255.0f);
  }
}

void
filter_run_gamma(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
	dim3 block(16, 16);
	gamma_kernel<<<grid_for(w, h, block), block>>>(d_in, d_out, w, h, p->gamma_val);
	cudaDeviceSynchronize();
}
