#!/usr/bin/env bash
# env-dump.sh — Capture all environment metadata for the test session
#
# Run once at the start. Redirect to your log:
#   ./test/env-dump.sh >> test/padding-results.log

echo "==== ENVIRONMENT ===="
echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "-- GPU --"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,pci.bus_id --format=csv 2>&1
echo ""

echo "-- CUDA --"
nvcc --version 2>&1 || echo "nvcc not found"
echo ""

echo "-- Runtime --"
echo "bun: $(bun --version 2>&1)"
echo "node-llama-cpp: $(bun -e 'import p from "node-llama-cpp/package.json" with { type: "json" }; console.log(p.version)' 2>&1)"
echo ""

echo "-- Model cache --"
ls -lh ~/.cache/qmd/models/ggml-org--embeddinggemma-300M-GGUF/ 2>&1 || echo "Model not cached!"
echo ""

echo "-- GPU processes --"
nvidia-smi --query-compute-apps=pid,name,used_gpu_memory --format=csv 2>&1 || echo "none"
echo ""

echo "-- System --"
uname -r
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "==== END ENVIRONMENT ===="
echo ""
