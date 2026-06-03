#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

timeout_seconds="${GOTENX_BENCHMARK_TIMEOUT:-300}"
destination="${GOTENX_XCODEBUILD_DESTINATION:-platform=macOS,arch=arm64}"
output_path="${GOTENX_BENCHMARK_OUTPUT:-$(pwd)/.build/benchmarks/latest.json}"
derived_data_path="${GOTENX_BENCHMARK_DERIVED_DATA:-$(pwd)/.build/xcodebuild-benchmarks}"
configuration="${GOTENX_BENCHMARK_CONFIGURATION:-Release}"
cell_count="${GOTENX_BENCHMARK_CELLS:-100}"
warmup_iterations="${GOTENX_BENCHMARK_WARMUP:-5}"
measured_iterations="${GOTENX_BENCHMARK_ITERATIONS:-30}"
newton_iterations="${GOTENX_BENCHMARK_NEWTON_ITERATIONS:-3}"

mkdir -p "$(dirname "${output_path}")"

perl -e 'alarm shift; exec @ARGV' "${timeout_seconds}" \
  xcodebuild -quiet build \
  -scheme GotenxBenchmarks \
  -destination "${destination}" \
  -configuration "${configuration}" \
  -derivedDataPath "${derived_data_path}" \
  CODE_SIGNING_ALLOWED=NO

binary_path="$(find "${derived_data_path}/Build/Products" -type f -name GotenxBenchmarks -perm -111 | head -n 1)"
if [[ -z "${binary_path}" ]]; then
  echo "GotenxBenchmarks binary not found under ${derived_data_path}/Build/Products" >&2
  exit 1
fi

"${binary_path}" \
  --cells "${cell_count}" \
  --warmup "${warmup_iterations}" \
  --iterations "${measured_iterations}" \
  --newton-iterations "${newton_iterations}" \
  --output "${output_path}"
