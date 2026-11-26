#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

////////////////////////////////////////////
// Filter table
//
//   X(enum_name, cli_name, params_help, run_fn)
//
// @note(gaureesh): to add a new filter, just add a new row
//                  in the table. then write its run function.

#define FILTER_TABLE(X) \
  X(FILTER_BLUR,                   "blur",       "-radius:<1..255> -sigma:<float>",     filter_run_blur)       \
  X(FILTER_KUWAHARA,               "kuwahara",   "-radius:<1..255> -sharpness:<float>", filter_run_kuwahara)   \
  X(FILTER_HISTOGRAM_EQUALIZATION, "histogram",  "",                                    filter_run_histogram)  \
  X(FILTER_BRIGHTNESS,             "brightness", "-brightness:<-255..255>",             filter_run_brightness) \
  X(FILTER_GAMMA,                  "gamma",      "-gamma:<0.1-5.0>",                    filter_run_gamma)

////////////////////////////////////////////
// Types

struct Image
{
	uint8_t *raw;
	int32_t width;
	int32_t height;
};

#define X(kind, ...) kind,
enum FilterKind
{
	FILTER_NONE,
	FILTER_TABLE(X)
	FILTER_COUNT,
};
#undef X

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

////////////////////////////////////////////
// Kernels

__device__ static inline int
clampi(int v, int lo, int hi)
{
	return v < lo ? lo : (v > hi ? hi : v);
}

#define MAX_RADIUS 255
#define BLOCK_X 32
#define BLOCK_Y 8

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

// blur vertical pass: reads the float intermediate from global memory (cached)
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

// mean per channel and an overall standard deviation for one rectangular
// region around (cx, cy). offsets x0..x1, y0..y1 are relative to the center
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
		// E[x^2] - E[x]^2 can dip slightly below zero from float rounding
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
		float w = powf(1.0f / ratio, sharpness);     // in (0, 1]
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

	// cumulative sum is done; find the first nonzero bin
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

////////////////////////////////////////////
// Per-filter run

static dim3
grid_for(int w, int h, dim3 block)
{
	return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
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

static void
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

static void
filter_run_kuwahara(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
	dim3 block(16, 16);
	kuwahara_kernel<<<grid_for(w, h, block), block>>>(d_in, d_out, w, h, p->radius, p->sharpness);
	cudaDeviceSynchronize();
}

static void
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

static void
filter_run_brightness(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
	dim3 block(16, 16);
	brightness_kernel<<<grid_for(w, h, block), block>>>(d_in, d_out, w, h, p->brightness);
	cudaDeviceSynchronize();
}

static void
filter_run_gamma(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
	dim3 block(16, 16);
	gamma_kernel<<<grid_for(w, h, block), block>>>(d_in, d_out, w, h, p->gamma_val);
	cudaDeviceSynchronize();
}

////////////////////////////////////////////
// Dispatch table (generated from FILTER_TABLE; rows are in enum order)

#define X(kind, name, help, run) { name, help, run },
static const
FilterDesc filter_descs[FILTER_COUNT] =
{
	{ "none", "", 0 },   // FILTER_NONE
	FILTER_TABLE(X)
};
#undef X

static FilterKind
filter_from_name(const char *name)
{
	for (int k = 1; k < FILTER_COUNT; ++k)
		if (strcmp(filter_descs[k].name, name) == 0) return (FilterKind)k;
	return FILTER_NONE;
}

////////////////////////////////////////////
// Host code

// if `arg` starts with `key`, returns pointer to the text after it, else NULL
static const char *opt_value(const char *arg, const char *key)
{
	size_t n = strlen(key);
	return strncmp(arg, key, n) == 0 ? arg + n : 0;
}

static void print_help(FILE *out)
{
	// @note(gaureesh) we can add more flags later, maybe control
	// tiling or some other kernel related data.

	fprintf(out, "usage:\n  img_filter <file_name> -filter:<filter> [options]\n\n"
	             "filters:\n");

	for (int k = 1; k < FILTER_COUNT; ++k)
	{
		fprintf(out, "  -filter:%-10s %s\n", filter_descs[k].name, filter_descs[k].params_help);
	}

	fprintf(out,
	        "\n"
	        "other options:\n"
	        "  -out:<path>   output png (default: out.png)\n");
}

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "usage error: img_filter <file_name> [options]\n"
		                "run `img_filter -help` to see list of options.\n");
		return 1;
	}

	const char *first_arg = argv[1];

	if (strcmp(first_arg, "-help") == 0)
	{
		print_help(stdout);
		return 0;
	}

	// parsing options
	FilterKind  kind     = FILTER_NONE;
	const char *out_path = "out.png";
	Params      params   = { 4, 2.0f, 8.0f, 0.0f, 1.0f };

	for (int i = 2; i < argc; ++i)
	{
		const char *opt = argv[i];
		const char *v;

		if ((v = opt_value(opt, "-filter:")))
		{
			kind = filter_from_name(v);
			if (kind == FILTER_NONE)
			{
				fprintf(stderr, "option error: unknown filter `%s`.\n", v);
				return 1;
			}
		}
		else if ((v = opt_value(opt, "-radius:")))     params.radius     = atoi(v);
		else if ((v = opt_value(opt, "-sigma:")))      params.sigma      = (float)atof(v);
		else if ((v = opt_value(opt, "-sharpness:")))  params.sharpness  = (float)atof(v);
		else if ((v = opt_value(opt, "-brightness:"))) params.brightness = (float)atof(v);
		else if ((v = opt_value(opt, "-gamma:")))      params.gamma_val  = (float)atof(v);
		else if ((v = opt_value(opt, "-out:")))        out_path          = v;
		else
		{
			fprintf(stderr, "option error: unknown option `%s`.\n", opt);
			return 1;
		}
	}

	if (kind == FILTER_NONE)
	{
		fprintf(stderr, "option error: no filter given, use -filter:<name>.\n");
		return 1;
	}
	if (params.radius < 1)   params.radius = 1;
	if (params.radius > 255) params.radius = 255;
	if (params.brightness < -255.0f) params.brightness = -255.0f;
	if (params.brightness >  255.0f) params.brightness =  255.0f;
	if (params.gamma_val < 0.1f) params.gamma_val = 0.1f;
	if (params.gamma_val > 5.0f) params.gamma_val = 5.0f;

	// loading the image
	Image img = {0};
	{
		const int32_t desired_channels = 3;

		int32_t channels;
		img.raw = stbi_load(first_arg, &img.width, &img.height, &channels, desired_channels);

		if (img.raw == 0)
		{
			fprintf(stderr, "load error: failed to load image `%s`.\n", first_arg);
			return 1;
		}
	}

	const FilterDesc *desc = &filter_descs[kind];

	size_t bytes = (size_t)img.width * img.height * 3;
	uint8_t *d_in = 0, *d_out = 0;

	cudaMalloc(&d_in,  bytes);
	cudaMalloc(&d_out, bytes);
	cudaMemcpy(d_in, img.raw, bytes, cudaMemcpyHostToDevice);

	desc->run(&params, d_in, d_out, img.width, img.height);

	uint8_t *result = (uint8_t *)malloc(bytes);
	cudaMemcpy(result, d_out, bytes, cudaMemcpyDeviceToHost);

	if (!stbi_write_png(out_path, img.width, img.height, 3, result, img.width * 3))
	{
		fprintf(stderr, "write error: failed to write `%s`.\n", out_path);
		return 1;
	}
	printf("%s: wrote `%s`\n", desc->name, out_path);

	free(result);
	cudaFree(d_in);
	cudaFree(d_out);
	stbi_image_free(img.raw);
	return 0;
}
