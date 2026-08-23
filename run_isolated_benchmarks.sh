#!/bin/bash
set -e

PYTHON_BIN="/home/aicloud/kuldeep/venvs/main/bin/python"

echo "=============================================================================="
echo "  Executing Process-Isolated Param2 Benchmark Suite (H200)"
echo "=============================================================================="

echo -e "\n>>> [1/3] Running Process 1: Torch-ParaS Backend (paras:1)..."
$PYTHON_BIN small_param2_benchmark_cuda_paras.py --backend paras

echo -e "\n>>> [2/3] Running Process 2: Native PyTorch CUDA (cuda:0)..."
$PYTHON_BIN small_param2_benchmark_cuda_paras.py --backend cuda

echo -e "\n>>> [3/3] Running Process 3: Generating Parity & Comparison Report..."
$PYTHON_BIN compare_param2_results.py

echo -e "\n=============================================================================="
echo "  All 3 Stages Completed Successfully!"
echo "  Report generated: PARAM2_H200_BENCHMARK_REPORT.md"
echo "=============================================================================="
