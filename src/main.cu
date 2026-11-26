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
//   X(enum_name, cli_name, params_help, make_fn, run_fn)
//
// @note(gaureesh): to add a new filter, just add a new row
//                  in the table. then write its make / run
//                  functions.

#define FILTER_TABLE(X) \
	X(FILTER_BLUR,     "blur",     "-radius:<1..255> -sigma:<float>",     filter_make_blur,     filter_run_blur)     \
	X(FILTER_KUWAHARA, "kuwahara", "-radius:<1..255> -sharpness:<float>", filter_make_kuwahara, filter_run_kuwahara) \
    X(FILTER_HISTOGRAM_EQUALIZATION, "histogram","", filter_make_histogram, filter_run_histogram) \
    X(FILTER_BRIGHTNESS, "brightness", "-value:<-255..0..255>", filter_make_brightness, filter_run_brightness)  \
    X(FILTER_GAMMA, "gamma" , "-gamma:<0.1-5.0>",filter_make_gamma, filter_run_gamma)
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

		struct
		{ 
            uint32_t * d_hist;
            uint8_t * d_final_vals;
            uint32_t total_pixels;
		} histogram;
		
        struct
		{
			float bright;
		} brightness;
        
        struct {
            float gamma_val;
        } gamma;
	};
};

// raw values from the command line, turned into a Filter by a make function
struct Params
{
	int   radius;
	float sigma;
	float sharpness;
    float brightness;
    float gamma_val;
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
	
	// todo: kuwahara
	int i = 3 * (y * width + x);
	out[i + 0] = in[i + 0];
	out[i + 1] = in[i + 1];
	out[i + 2] = in[i + 2];
}

__global__ void 
histogram_kernel(const uint8_t *in, uint32_t *d_hist, int width, int height, int total_pixels)
{
    __shared__ uint32_t hist[256]; 

    int tid =threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block=blockDim.x * blockDim.y;
    
    for(int c=tid;c<256;c+=threads_per_block){
        hist[c]=0;
    }
    __syncthreads();

    //hist[i]=0 for all i
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// todo: kuwahara
    if( x < width && y<height) {
	int i = 3 * (y * width + x);
	uint8_t r= in[i + 0];
	uint8_t g = in[i + 1];
	uint8_t b = in[i + 2];

    float z=(0.299f * r + 0.587f * g + 0.114f * b + 0.5f);
    uint32_t val= (z > 255.0f) ? 255u : (uint32_t) z;

    atomicAdd(&hist[val],1u);
    }
    __syncthreads();
    //histogram will be filled up with values

    
    for(int c=tid;c<256;c+=threads_per_block){
        atomicAdd(&d_hist[c],hist[c]);
    }


}

__global__ void 
cdf_kernel(const uint32_t *d_hist,uint8_t * d_fval, int width, int height, int total_pixels)
{
    __shared__ uint32_t cdf[256]; 
    __shared__ uint32_t warps_total[8];
    __shared__ uint32_t cdf_min;

    if(threadIdx.x>=256) return;
    int tid=threadIdx.x;

    int lane=tid&31;
    int warp_id=tid >>5;

    uint32_t sum=d_hist[tid];
    for(int i=1;i<32;i*=2){
        uint32_t x=__shfl_up_sync(0xffffffff,sum,i);
        if(lane >=i) sum+=x;
    }
    __syncthreads();
    
    if(lane==31) warps_total[warp_id]=sum;

    __syncthreads();

    if(warp_id==0){
        uint32_t warp_sum=0;
        if(lane<8) warp_sum=warps_total[lane];

        for(int i=1;i<8;i*=2){
            uint32_t x=__shfl_up_sync(0xffffffff,warp_sum,i);
            if(lane >=i && lane<8) warp_sum+=x;
        }

        if(lane<8) warps_total[lane]=warp_sum;

        }
   __syncthreads();

   if(warp_id>0) sum+=warps_total[warp_id-1];

   cdf[tid]=sum;

    __syncthreads();
   //cummulative sum is calculated

   if(tid==0){
       for(int i=0;i<256;i++){
           if(cdf[i]>0) {
               cdf_min=cdf[i];
               break;
               }
       }
   }
   __syncthreads();
   
   float num=sum-cdf_min;
   float den=total_pixels-cdf_min;
   float val=0;
   if(den!=0 ) val=(num/den) * 255.0f + 0.5f;
   d_fval[tid]=(uint8_t) val;


}
__global__ void 
writeback_histogram_kernel(const uint8_t *in, uint8_t *out,uint8_t *d_fval, int w, int h, int total_pixels)
{
    
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x >=w || y>=h) return;
	
    int idx = 3 * (y * w + x);
	uint8_t r= in[idx + 0];
	uint8_t g = in[idx + 1];
	uint8_t b = in[idx + 2];

    float z=(0.299f * r + 0.587f * g + 0.114f * b + 0.5f);
    uint32_t val= (z > 255.0f) ? 255u : (uint32_t) z;

    float new_val=d_fval[val];
    float scale=1;
    if(z!=0.5f) scale=new_val/(z-0.5f);

    for(int i=0;i<3;i++){
        float v;
        uint8_t update;
        if(i==0) v= r*scale;
        if(i==1) v= g*scale;
        if(i==2) v= b*scale;
    
        v+=0.5f;
        if(v<=0.0f) update=0;
        else if(v>=255.0f) update=255;
        else update=(uint8_t) v;
        out[idx+i]=update;
    }
}
__global__ void 
brightness_kernel(const uint8_t *in, uint8_t *out, int width, int height, float brightness)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	
	int i = 3 * (y * width + x);
    for(int j=0;j<3;j++){
	    int val = in[i + j] + brightness;
        val=max(0,min(255,val));
        out[i+j]=(uint8_t)val;
    }
}

__global__ void
gamma_kernel(const uint8_t *in, uint8_t *out, int width, int height, float gamma)
	{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	int i = 3*(y * width + x);
    for(int j=0; j<3;j++){
	    float normal_val= in[i+j]/255.0f;
        float correction=powf(normal_val,gamma);
        out[i+j]=(uint8_t) (correction * 255.0f);
    }
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

static Filter 
filter_make_histogram(const Params *p)
{
	Filter f; memset(&f, 0, sizeof f);
    cudaMalloc(&f.histogram.d_hist,256 * sizeof(uint32_t));
    cudaMalloc(&f.histogram.d_final_vals,256 * sizeof(uint8_t));
    cudaMemset(f.histogram.d_hist,0,256 * sizeof(uint32_t));
    return f;
}

static Filter
filter_make_brightness(const Params *p)
{
	Filter f; memset(&f, 0, sizeof f);
	f.kind = FILTER_BRIGHTNESS;
	f.brightness.bright = p->brightness;
	return f;
}

static Filter filter_make_gamma(const Params *p){
    Filter f;
	memset(&f, 0, sizeof f);
    f.kind = FILTER_GAMMA;
    f.gamma.gamma_val=p->gamma_val;
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

static void 
filter_run_histogram(const Filter *f, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
	dim3 block(16, 16);
    int total_pixels=w*h;
	histogram_kernel<<<grid_for(w, h, block), block>>>(d_in, f->histogram.d_hist, w, h, total_pixels);
	cdf_kernel<<<1,256>>>(f->histogram.d_hist, f->histogram.d_final_vals, w, h, total_pixels);
	writeback_histogram_kernel<<<grid_for(w, h, block), block>>>(d_in,d_out,f->histogram.d_final_vals, w, h, total_pixels);
	cudaDeviceSynchronize();
}

static void
filter_run_brightness(const Filter *f, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
	dim3 block(16, 16);
	brightness_kernel<<<grid_for(w, h, block), block>>>(d_in, d_out, w, h, f->brightness.bright);
    cudaDeviceSynchronize();
}

static void
filter_run_gamma(const Filter *f, const uint8_t *d_in, uint8_t *d_out, int w, int h)
{
	dim3 block(16, 16);
	gamma_kernel<<<grid_for(w, h, block), block>>>(d_in, d_out, w, h, f->gamma.gamma_val);
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
        else if ((v = opt_value(opt, "-gamma:"))) params.gamma_val=(float)atof(v);
		else if ((v = opt_value(opt, "-out:")))       out_path         = v;
		else if ((v = opt_value(opt, "-brightness:")))     params.brightness     = (float)atof(v);
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
	if (params.gamma_val < 0.1) params.gamma_val=0.1;
    if (params.gamma_val > 5.0) params.gamma_val=5.0;

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
