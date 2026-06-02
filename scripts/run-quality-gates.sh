#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

timeout_seconds="${GOTENX_XCODEBUILD_TIMEOUT:-180}"
destination="${GOTENX_XCODEBUILD_DESTINATION:-platform=macOS,arch=arm64}"

perl -e 'alarm shift; exec @ARGV' "${timeout_seconds}" \
  xcodebuild -quiet test \
  -scheme swift-gotenx-Package \
  -destination "${destination}" \
  -only-testing:GotenxTests/RepositoryQualityGateTests \
  -only-testing:GotenxTests/GotenxConfigReaderTests \
  -only-testing:GotenxTests/TransportParameterValidationTests \
  -only-testing:GotenxTests/ConfigurationPriorityTests \
  -only-testing:GotenxTests/ConfigurationLoaderTests \
  -only-testing:GotenxTests/DerivedQuantitiesComputerTests \
  -only-testing:GotenxTests/SimulationRunnerEnergyDiagnosticsExperimentTests \
  -only-testing:GotenxTests/NumericalValidationTests \
  -only-testing:GotenxTests/Block1DCoeffsBuilderTests \
  -only-testing:GotenxTests/ProfileValidationMatrixTests \
  -only-testing:GotenxTests/ProfileValidationMatrixPerformanceTests
