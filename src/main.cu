#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

////////////////////////////////////////////
// Filter table
//
//   X(enum_name, cli_name, params_help, make_fn, run_fn)
//
// @note(gaureesh): to add a new filter, just add a new row
//                  in the table. then write its make / run
//                  functions.

#define FILTER_TABLE(X) \
	X(FILTER_BLUR,     "blur",     "-radius:<1..255> -sigma:<float>",     filter_make_blur,     filter_run_blur)     \
	X(FILTER_KUWAHARA, "kuwahara", "-radius:<1..255> -sharpness:<float>", filter_make_kuwahara, filter_run_kuwahara)

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

struct Filter
{
	FilterKind kind;
	
	union
	{
		struct
		{
			uint8_t radius;
			float sigma;
		} blur;
		
		struct
		{
			uint8_t radius;
			float sharpness;
		} kuwahara;
	};
};

// raw values from the command line, turned into a Filter by a make function
struct Params
{
	int   radius;
	float sigma;
	float sharpness;
};

typedef Filter (*FilterMakeFn)(const Params *p);
typedef void   (*FilterRunFn)(const Filter *f, const uint8_t *d_in, uint8_t *d_out, int w, int h);

struct FilterDesc
{
	const char  *name;
	const char  *params_help;
	FilterMakeFn make;
	FilterRunFn  run;
};

////////////////////////////////////////////
// Kernels (stubs: passthrough for now)

__global__ void 
blur_kernel(const uint8_t *in, uint8_t *out, int width, int height, int radius, float sigma)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	
	// TODO: blur
	int i = 3 * (y * width + x);
	out[i + 0] = in[i + 0];
	out[i + 1] = in[i + 1];
	out[i + 2] = in[i + 2];
}

__global__ void 
kuwahara_kernel(const uint8_t *in, uint8_t *out, int width, int height, int radius, float sharpness)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	
	// TODO: kuwahara
	int i = 3 * (y * width + x);
	out[i + 0] = in[i + 0];
	out[i + 1] = in[i + 1];
	out[i + 2] = in[i + 2];
}

////////////////////////////////////////////
// Per-filter make / run

static Filter 
filter_make_blur(const Params *p)
{
	Filter f; memset(&f, 0, sizeof f);
	f.kind = FILTER_BLUR;
	f.blur.radius = (uint8_t)p->radius;
	f.blur.sigma  = p->sigma;
	return f;
}

static Filter 
filter_make_kuwahara(const Params *p)
{
	Filter f; memset(&f, 0, sizeof f);
	f.kind = FILTER_KUWAHARA;
	f.kuwahara.radius    = (uint8_t)p->radius;
	f.kuwahara.sharpness = p->sharpness;
	return f;
}

static dim3 
grid_for(int w, int h, dim3 block)
{
	return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

static void 
filter_run_blur(const Filter *f, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
	dim3 block(16, 16);
	blur_kernel<<<grid_for(w, h, block), block>>>(d_in, d_out, w, h, f->blur.radius, f->blur.sigma);
	cudaDeviceSynchronize();
}

static void 
filter_run_kuwahara(const Filter *f, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
	dim3 block(16, 16);
	kuwahara_kernel<<<grid_for(w, h, block), block>>>(d_in, d_out, w, h, f->kuwahara.radius, f->kuwahara.sharpness);
	cudaDeviceSynchronize();
}

////////////////////////////////////////////
// Dispatch table (generated from FILTER_TABLE; rows are in enum order)

#define X(kind, name, help, make, run) { name, help, make, run },
static const 
FilterDesc filter_descs[FILTER_COUNT] =
{
	{ "none", "", 0, 0 },   // FILTER_NONE
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

	fprintf( out,
	         "\n"
	         "other options:\n"
	         "  -out:<path>   output png (default: out.png)\n" );
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
	Params      params   = { 4, 2.0f, 8.0f };
	
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
		else if ((v = opt_value(opt, "-radius:")))    params.radius    = atoi(v);
		else if ((v = opt_value(opt, "-sigma:")))     params.sigma     = (float)atof(v);
		else if ((v = opt_value(opt, "-sharpness:"))) params.sharpness = (float)atof(v);
		else if ((v = opt_value(opt, "-out:")))       out_path         = v;
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
	Filter filter = desc->make(&params);
	
	size_t bytes = (size_t)img.width * img.height * 3;
	uint8_t *d_in = 0, *d_out = 0;

	cudaMalloc(&d_in,  bytes);
	cudaMalloc(&d_out, bytes);
	cudaMemcpy(d_in, img.raw, bytes, cudaMemcpyHostToDevice);
	
	desc->run(&filter, d_in, d_out, img.width, img.height);
	
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
