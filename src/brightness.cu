#include "filters.h"

__global__ void
brightness_kernel(const uint8_t *in, uint8_t *out, int width, int height, float brightness)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int i = 3 * (y * width + x);
  for (int j = 0; j < 3; j++)
  {
    int val = in[i + j] + brightness;
    val = max(0, min(255, val));
    out[i + j] = (uint8_t)val;
  }
}

void
filter_run_brightness(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
	dim3 block(16, 16);
	brightness_kernel<<<grid_for(w, h, block), block>>>(d_in, d_out, w, h, p->brightness);
	cudaDeviceSynchronize();
}
