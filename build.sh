#!/bin/bash

mkdir -p bin

nvcc ./src/main.cu -o ./bin/img_filter -I ./vendor -diag-suppress=550
