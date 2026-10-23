# cxx/ Makefile - single nvcc build for the deployment binaries.
#
# SM arch is auto-detected via nvidia-smi; override with `make SM=sm_87`.
# Known-good pairings: sm_87 Jetson AGX Orin (JetPack 6.2 / TRT 10.3),
# sm_86 A6000 (TRT 10.16), sm_89 4090 Laptop (dev box).
#
# runner: the 31-segment GPU-resident chain (links TensorRT).
# render_bench / render_matrix: rendering benchmarks; compile the vendored
#   DGR rasterizer sources directly (no Python extension build needed).
# cost_volume_bench: standalone cost-volume kernel microbenchmark.

CUDA_DIR ?= /usr/local/cuda
NVCC ?= $(CUDA_DIR)/bin/nvcc
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
DGR := $(ROOT)/third_party/diff-gaussian-rasterization-modified

# runner links TensorRT. Resolve BOTH the include dir and the lib dir from
# the trtexec on PATH: (a) engines must be deserialized by the SAME TRT that
# built them - mixing serializations (e.g. system TRT 8.6 vs trtexec TRT 10.4
# on a dev box) fails deserialization with the stdArchiveReader version-tag
# assert; (b) the NvInfer headers must match the lib too - compiling against
# a system TRT 10.9 header while linking a 10.4 lib shifts every vtable slot
# and crashes virtual calls at the first getNbIOTensors/createExecutionContext.
TRTEXEC := $(shell command -v trtexec 2>/dev/null)
ifeq ($(TRTEXEC),)
# Jetson devkits install trtexec under /usr/src/tensorrt (not on PATH).
TRTEXEC := $(shell for c in /usr/src/tensorrt/bin/trtexec /usr/src/TensorRT-*/bin/trtexec; do [ -x "$$c" ] && { echo "$$c"; break; }; done)
endif
TRT_HOME ?= $(shell d=$$(readlink -f $(TRTEXEC)); echo $${d%/*/*})
# Tarball installs keep headers/libs under TRT_HOME; JetPack deb installs put
# trtexec under /usr/src/tensorrt but headers/libs in the multiarch dirs.
TRIPLET := $(shell cc -dumpmachine)
INC_CANDIDATES := $(TRT_HOME)/include/NvInfer.h \
                  $(TRT_HOME)/targets/*/include/NvInfer.h \
                  /usr/include/$(TRIPLET)/NvInfer.h
LIB_CANDIDATES := $(TRT_HOME)/lib/libnvinfer.so \
                  $(TRT_HOME)/targets/*/lib/libnvinfer.so \
                  /usr/lib/$(TRIPLET)/libnvinfer.so \
                  /usr/lib/$(TRIPLET)/libnvinfer.so.10
TRT_INC_DIR ?= $(patsubst %/NvInfer.h,%,$(firstword $(wildcard $(INC_CANDIDATES))))
TRT_LIB_DIR ?= $(patsubst %/libnvinfer.so,%,$(patsubst %/libnvinfer.so.10,%,$(firstword $(wildcard $(LIB_CANDIDATES)))))
ifeq ($(wildcard $(TRT_LIB_DIR)/libnvinfer.so*),)
$(error TRT_LIB_DIR "$(TRT_LIB_DIR)" holds no libnvinfer - point TRT_LIB_DIR at the TensorRT lib dir that matches the trtexec used for engine builds)
endif
ifeq ($(wildcard $(TRT_INC_DIR)/NvInfer.h),)
$(error TRT_INC_DIR "$(TRT_INC_DIR)" holds no NvInfer.h - point TRT_INC_DIR at the TensorRT include dir that matches the trtexec used for engine builds)
endif

CC_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
SM ?= sm_$(CC_CAP)

NVCCFLAGS := -O3 -std=c++17 -arch=$(SM)
DGR_SRC := $(DGR)/cuda_rasterizer/forward.cu $(DGR)/cuda_rasterizer/backward.cu $(DGR)/cuda_rasterizer/rasterizer_impl.cu
DGR_INC := -I$(DGR)/cuda_rasterizer -I$(DGR)/third_party/glm

.PHONY: all runner render_bench render_matrix cost_volume_bench clean
all: runner render_bench render_matrix cost_volume_bench

runner: $(ROOT)/cxx/runner.cu $(ROOT)/cxx/flash_attn.cu $(ROOT)/cxx/so3_const.cuh
	$(NVCC) $(NVCCFLAGS) -I$(ROOT)/cxx -I$(TRT_INC_DIR) -L$(TRT_LIB_DIR) -Xlinker -rpath=$(TRT_LIB_DIR) -lnvinfer -o $(ROOT)/runner $(ROOT)/cxx/runner.cu

render_bench: $(ROOT)/cxx/render_bench.cu $(DGR_SRC)
	$(NVCC) $(NVCCFLAGS) $(DGR_INC) -o $(ROOT)/render_bench $(ROOT)/cxx/render_bench.cu $(DGR_SRC)

render_matrix: $(ROOT)/cxx/render_matrix.cu $(DGR_SRC)
	$(NVCC) $(NVCCFLAGS) $(DGR_INC) -o $(ROOT)/render_matrix $(ROOT)/cxx/render_matrix.cu $(DGR_SRC)

cost_volume_bench: $(ROOT)/cxx/cost_volume_bench.cu
	$(NVCC) $(NVCCFLAGS) -o $(ROOT)/cost_volume_bench $(ROOT)/cxx/cost_volume_bench.cu

clean:
	rm -f $(ROOT)/runner $(ROOT)/render_bench $(ROOT)/render_matrix $(ROOT)/cost_volume_bench
