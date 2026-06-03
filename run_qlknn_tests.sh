#!/bin/bash

# QLKNN test runner.
# Sets up Python environment for PythonKit before running tests.

set -euo pipefail

# Configure Python library path for PythonKit
export PYTHON_LIBRARY="/Library/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib"
export PYTHONPATH="/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages"

# Verify Python setup
echo "Python configuration:"
echo "  PYTHON_LIBRARY: $PYTHON_LIBRARY"
echo "  PYTHONPATH: $PYTHONPATH"
echo ""

# Verify fusion_surrogates is installed
if python3 -c "import fusion_surrogates" 2>/dev/null; then
    echo "fusion_surrogates is installed"
else
    echo "fusion_surrogates is NOT installed"
    echo ""
    echo "Please install fusion_surrogates:"
    echo "  pip install fusion-surrogates"
    exit 1
fi

echo ""
echo "Running QLKNN tests..."
echo ""

# Run QLKNN tests with environment variables.
timeout_seconds="${GOTENX_XCODEBUILD_TIMEOUT:-180}"
destination="${GOTENX_XCODEBUILD_DESTINATION:-platform=macOS,arch=arm64}"

perl -e 'alarm shift; exec @ARGV' "${timeout_seconds}" \
  xcodebuild -quiet test \
  -scheme swift-gotenx-Package \
  -destination "${destination}" \
  -only-testing:GotenxTests/QLKNNTransportModelTests

echo ""
echo "Tests completed"
