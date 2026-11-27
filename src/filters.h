#pragma once
#include "common.h"

////////////////////////////////////////////
// Run-function declarations (defined in the .cu files)

void filter_run_blur(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h);
void filter_run_kuwahara(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h);
void filter_run_histogram(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h);
void filter_run_brightness(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h);
void filter_run_gamma(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h);
void filter_run_median(const Params *p, const uint8_t *d_in, uint8_t *d_out, int w, int h);

////////////////////////////////////////////
// Filter table
//
//   X(enum_name, cli_name, params_help, run_fn)
//
// to add a new filter, add a row here, declare its run function above,
// and write it in its own .cu file.

#define FILTER_TABLE(X) \
  X(FILTER_BLUR,                   "blur",       "-radius:<1..255> -sigma:<float>",     filter_run_blur)       \
  X(FILTER_KUWAHARA,               "kuwahara",   "-radius:<1..255> -sharpness:<float>", filter_run_kuwahara)   \
  X(FILTER_HISTOGRAM_EQUALIZATION, "histogram",  "",                                    filter_run_histogram)  \
  X(FILTER_BRIGHTNESS,             "brightness", "-brightness:<-255..255>",             filter_run_brightness) \
  X(FILTER_GAMMA,                  "gamma",      "-gamma:<0.1-5.0>",                    filter_run_gamma)      \
  X(FILTER_MEDIAN,                 "median",     "-radius:<1..255>",                    filter_run_median) 

#define X(kind, ...) kind,
enum FilterKind
{
  FILTER_NONE,
  FILTER_TABLE(X)
  FILTER_COUNT,
};
#undef X
