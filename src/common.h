#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#define MAX_RADIUS 255
#define BLOCK_X 32
#define BLOCK_Y 8

// raw values from the command line, passed straight to the run functions
struct Params
{
  int   radius;
  float sigma;
  float sharpness;
  float brightness;
  float gamma_val;
};

typedef void (*FilterRunFn)(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h);

struct FilterDesc
{
  const char  *name;
  const char  *params_help;
  FilterRunFn  run;
};

  __host__ __device__ static inline int
clampi(int v, int lo, int hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}

  static inline dim3
grid_for(int w, int h, dim3 block)
{
  return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}
