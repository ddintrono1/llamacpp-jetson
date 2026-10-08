# llama.cpp for Jetson AGX Orin on JetPack 7.2 (L4T R39.2, Ubuntu 24.04, CUDA 13.2)
# The project is split into two images: one required for building and the other for runtime only.
# The build image can be removed if no further modifications are going to be performed.
FROM nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04 AS build

ARG JOBS=4

RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN git clone --depth 1 --branch v0.6.0 https://github.com/ggml-org/llama.cpp .


################################
# CMake Options
################################
# GGML_CUDA=ON                 -> CUDA backend
# CMAKE_CUDA_ARCHITECTURES=87  -> kernels only for Orin's GPU (sm_87)
# BUILD_SHARED_LIBS=OFF        -> libllama/libggml linked statically into each binary,
#                                 so the runtime stage needs only the executables
# LLAMA_CURL=OFF               -> no libcurl (no model download; mount models instead)
#                                 (option name may differ in newer versions: check CMake output)
# --allow-shlib-undefined      -> the real libcuda.so (driver) is not in the image during
#                                 `docker build`; the NVIDIA container runtime mounts it from
#                                 the host at `docker run`. This lets the link step succeed.
RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=87 \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build --config Release -j${JOBS} \
        --target llama-server llama-cli llama-bench

# -------------------------------------------------------------- runtime stage
FROM nvcr.io/nvidia/cuda:13.2.0-runtime-ubuntu24.04

# libgomp1: OpenMP runtime used by ggml's CPU threads.
# Remove the CUDA forward-compatibility libraries (compat/libcuda.so*).
# They are only needed when the container's CUDA is newer than the host driver, which is not the case here (both 13.2).
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/cuda*/compat* && \
    ldconfig


COPY --from=build /src/build/bin/llama-server \
                  /src/build/bin/llama-cli \
                  /src/build/bin/llama-bench \
                  /usr/local/bin/

EXPOSE 8080




