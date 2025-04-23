#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export CUDA_INJECTION64_PATH=${SCRIPT_DIR}/cupti_sandbox/libcupti_trace_injection.so
export NVTX_INJECTION64_PATH=/seeshpool/calebs_hidey_hole/cuda/targets/x86_64-linux/lib/libcupti.sowh

echo ${CUDA_INJECTION64_PATH}