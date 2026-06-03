import Foundation
import MLX
import Logging
import GotenxCore
import GotenxPhysics

@main
enum BenchmarkCommand {
    static func main() throws {
        BenchmarkLogging.bootstrap()

        let options = try BenchmarkOptions(arguments: Array(CommandLine.arguments.dropFirst()))
        if options.showsHelp {
            print(BenchmarkOptions.helpText)
            return
        }

        let harness = try RuntimeBenchmarkHarness(options: options)
        let metrics = try [
            harness.measureSolverSourceEvaluation(),
            harness.measureDiagnosticSourceEvaluation(),
            harness.measureCoefficientAssemblyWithoutSources(),
            harness.measureCoefficientAssembly(),
            harness.measureNewtonStep(),
            harness.measureNewtonStepDenseLU(),
            harness.measureNewtonStepBandedJacobianCandidate(),
            harness.measureNewtonStepWithSingleRefinement(),
            harness.measureNewtonStepWithLegacyRefinement(),
            harness.measureNewtonStepMetalCandidate()
        ]

        let report = RuntimeBenchmarkReport(
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            configuration: options.reportConfiguration,
            metrics: metrics
        )
        try report.write(to: options.outputURL)

        print(report.summary)
        print("Benchmark results: \(options.outputURL.path)")
    }
}

private enum BenchmarkLogging {
    private static let bootstrapOnce: Void = {
        let rawLevel = ProcessInfo.processInfo.environment["GOTENX_BENCHMARK_LOG_LEVEL"] ?? "critical"
        let level = Logger.Level(rawValue: rawLevel.lowercased()) ?? .critical
        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardError(label: label)
            handler.logLevel = level
            return handler
        }
    }()

    static func bootstrap() {
        _ = bootstrapOnce
    }
}

private struct BenchmarkOptions {
    let cellCount: Int
    let warmupIterations: Int
    let measuredIterations: Int
    let newtonIterations: Int
    let outputURL: URL
    let showsHelp: Bool

    var reportConfiguration: RuntimeBenchmarkReport.Configuration {
        RuntimeBenchmarkReport.Configuration(
            cellCount: cellCount,
            warmupIterations: warmupIterations,
            measuredIterations: measuredIterations,
            newtonIterations: newtonIterations
        )
    }

    init(arguments: [String]) throws {
        var cellCount = 100
        var warmupIterations = 5
        var measuredIterations = 30
        var newtonIterations = 3
        var outputURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/benchmarks/latest.json")
        var showsHelp = false

        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--help", "-h":
                showsHelp = true
                index += 1
            case "--cells":
                cellCount = try Self.integerValue(after: argument, in: arguments, at: &index)
            case "--warmup":
                warmupIterations = try Self.integerValue(after: argument, in: arguments, at: &index)
            case "--iterations":
                measuredIterations = try Self.integerValue(after: argument, in: arguments, at: &index)
            case "--newton-iterations":
                newtonIterations = try Self.integerValue(after: argument, in: arguments, at: &index)
            case "--output":
                outputURL = URL(fileURLWithPath: try Self.stringValue(after: argument, in: arguments, at: &index))
            default:
                throw BenchmarkError.unknownArgument(argument)
            }
        }

        guard cellCount >= 8 else {
            throw BenchmarkError.invalidValue("--cells must be at least 8")
        }
        guard warmupIterations >= 0 else {
            throw BenchmarkError.invalidValue("--warmup must be non-negative")
        }
        guard measuredIterations > 0 else {
            throw BenchmarkError.invalidValue("--iterations must be positive")
        }
        guard newtonIterations > 0 else {
            throw BenchmarkError.invalidValue("--newton-iterations must be positive")
        }

        self.cellCount = cellCount
        self.warmupIterations = warmupIterations
        self.measuredIterations = measuredIterations
        self.newtonIterations = newtonIterations
        self.outputURL = outputURL
        self.showsHelp = showsHelp
    }

    private static func integerValue(
        after argument: String,
        in arguments: [String],
        at index: inout Int
    ) throws -> Int {
        let stringValue = try stringValue(after: argument, in: arguments, at: &index)
        guard let value = Int(stringValue) else {
            throw BenchmarkError.invalidValue("\(argument) expects an integer")
        }
        return value
    }

    private static func stringValue(
        after argument: String,
        in arguments: [String],
        at index: inout Int
    ) throws -> String {
        let valueIndex = index + 1
        guard valueIndex < arguments.count else {
            throw BenchmarkError.missingValue(argument)
        }
        let value = arguments[valueIndex]
        guard !value.hasPrefix("--") else {
            throw BenchmarkError.missingValue(argument)
        }
        index += 2
        return value
    }

    static let helpText = """
    Usage: GotenxBenchmarks [options]

    Options:
      --cells <n>               Number of radial cells. Default: 100
      --warmup <n>              Warmup iterations before measurement. Default: 5
      --iterations <n>          Measured iterations. Default: 30
      --newton-iterations <n>   Maximum Newton iterations per step. Default: 3
      --output <path>           JSON output path. Default: .build/benchmarks/latest.json
    """
}

private enum BenchmarkError: Error, CustomStringConvertible {
    case unknownArgument(String)
    case missingValue(String)
    case invalidValue(String)

    var description: String {
        switch self {
        case .unknownArgument(let argument):
            "Unknown argument: \(argument)"
        case .missingValue(let argument):
            "Missing value for \(argument)"
        case .invalidValue(let message):
            message
        }
    }
}

private struct RuntimeBenchmarkHarness {
    let options: BenchmarkOptions
    let mesh: MeshConfig
    let geometry: Geometry
    let geometricFactors: GeometricFactors
    let profiles: CoreProfiles
    let staticParameters: StaticRuntimeParameters
    let newtonStaticParameters: StaticRuntimeParameters
    let dynamicParameters: DynamicRuntimeParameters
    let transportModel: ConstantTransportModel
    let sourceModel: CompositeSourceModel
    let sourceParameters: SourceParameters

    init(options: BenchmarkOptions) throws {
        self.options = options
        self.mesh = MeshConfig(
            cellCount: options.cellCount,
            majorRadius: 6.2,
            minorRadius: 2.0,
            toroidalField: 5.3,
            geometryType: .circular
        )
        self.geometry = Geometry(config: mesh)
        self.geometricFactors = GeometricFactors.from(
            geometry: geometry,
            evaluationMode: .deferred
        )
        self.profiles = Self.makeProfiles(cellCount: options.cellCount)
        self.staticParameters = StaticRuntimeParameters(
            mesh: mesh,
            evolveIonHeat: true,
            evolveElectronHeat: true,
            evolveElectronDensity: true,
            evolvePoloidalFlux: true,
            solverType: .newtonRaphson,
            theta: 1.0,
            solverTolerance: 1e-5,
            solverMaximumIterations: options.newtonIterations
        )
        self.newtonStaticParameters = StaticRuntimeParameters(
            mesh: mesh,
            evolveIonHeat: true,
            evolveElectronHeat: true,
            evolveElectronDensity: false,
            evolvePoloidalFlux: false,
            solverType: .newtonRaphson,
            theta: 1.0,
            solverTolerance: 1e-5,
            solverMaximumIterations: options.newtonIterations
        )
        self.dynamicParameters = try Self.makeDynamicParameters()
        self.transportModel = ConstantTransportModel(
            ionHeatDiffusivity: 1.0,
            electronHeatDiffusivity: 1.0,
            particleDiffusivity: 0.3,
            convectionVelocity: 0.0
        )
        self.sourceModel = CompositeSourceModel(sources: [
            "ohmic": OhmicHeatingSource(),
            "fusion": FusionPowerSource(),
            "ionElectronExchange": IonElectronExchangeSource(),
            "bremsstrahlung": BremsstrahlungSource(),
            "ecrh": ECRHSource(),
            "gasPuff": GasPuffSource(),
            "impurityRadiation": ImpurityRadiationSource()
        ])
        self.sourceParameters = SourceParameters(modelType: "composite", parameters: [:])
    }

    func measureSolverSourceEvaluation() throws -> RuntimeBenchmarkMetric {
        try measure(name: "source_evaluation_solver_deferred") {
            let terms = try sourceModel.computeTerms(
                in: SourceEvaluationContext(
                    profiles: profiles,
                    geometry: geometry,
                    geometricFactors: geometricFactors,
                    parameters: sourceParameters,
                    purpose: .solver
                )
            )
            evaluate(terms)
        }
    }

    func measureDiagnosticSourceEvaluation() throws -> RuntimeBenchmarkMetric {
        try measure(name: "source_evaluation_diagnostic_eager") {
            let terms = try sourceModel.computeTerms(
                in: SourceEvaluationContext(
                    profiles: profiles,
                    geometry: geometry,
                    parameters: sourceParameters,
                    purpose: .diagnostic
                )
            )
            evaluate(terms)
        }
    }

    func measureCoefficientAssembly() throws -> RuntimeBenchmarkMetric {
        try measure(name: "coefficient_assembly_deferred") {
            let coeffs = try makeCoefficients(for: profiles)
            evaluate(coeffs)
        }
    }

    func measureCoefficientAssemblyWithoutSources() throws -> RuntimeBenchmarkMetric {
        try measure(name: "coefficient_assembly_without_sources") {
            let sources = SourceTerms.zero(
                cellCount: profiles.ionTemperature.shape[0],
                evaluationMode: .deferred,
                metadata: nil,
                validateDebugUnits: false
            )
            let coeffs = try makeCoefficients(for: profiles, sourceOverride: sources)
            evaluate(coeffs)
        }
    }

    func measureNewtonStep() throws -> RuntimeBenchmarkMetric {
        try measureNewtonStep(
            name: "newton_step_block_tridiagonal_limited_iterations",
            linearSolver: HybridLinearSolver()
        )
    }

    func measureNewtonStepDenseLU() throws -> RuntimeBenchmarkMetric {
        try measureNewtonStep(
            name: "newton_step_dense_lu_limited_iterations",
            linearSolver: HybridLinearSolver(usesBlockTridiagonalCandidate: false),
            usesBandedJacobianCandidate: false
        )
    }

    func measureNewtonStepBandedJacobianCandidate() throws -> RuntimeBenchmarkMetric {
        try measureNewtonStep(
            name: "newton_step_banded_jacobian_candidate_limited_iterations",
            linearSolver: HybridLinearSolver(),
            usesBandedJacobianCandidate: true
        )
    }

    func measureNewtonStepWithSingleRefinement() throws -> RuntimeBenchmarkMetric {
        try measureNewtonStep(
            name: "newton_step_single_refinement_limited_iterations",
            linearSolver: HybridLinearSolver(
                cpuRefinementIterations: 1,
                usesBlockTridiagonalCandidate: false
            ),
            usesBandedJacobianCandidate: false
        )
    }

    func measureNewtonStepWithLegacyRefinement() throws -> RuntimeBenchmarkMetric {
        try measureNewtonStep(
            name: "newton_step_legacy_refinement_limited_iterations",
            linearSolver: HybridLinearSolver(
                cpuRefinementIterations: 2,
                usesBlockTridiagonalCandidate: false
            ),
            usesBandedJacobianCandidate: false
        )
    }

    func measureNewtonStepMetalCandidate() throws -> RuntimeBenchmarkMetric {
        try measureNewtonStep(
            name: "newton_step_metal_cgnr_candidate_limited_iterations",
            linearSolver: HybridLinearSolver(
                gpuMaxIterations: 64,
                gpuDimensionLimit: 1024,
                usesBlockTridiagonalCandidate: false
            ),
            usesBandedJacobianCandidate: false
        )
    }

    private func measureNewtonStep(
        name: String,
        linearSolver: HybridLinearSolver,
        usesBandedJacobianCandidate: Bool = false
    ) throws -> RuntimeBenchmarkMetric {
        try measure(name: name) {
            let solver = NewtonRaphsonSolver(
                tolerance: newtonStaticParameters.solverTolerance,
                maximumIterations: options.newtonIterations,
                theta: newtonStaticParameters.theta,
                linearSolver: linearSolver,
                usesBandedJacobianCandidate: usesBandedJacobianCandidate
            )

            let result = solver.solve(
                timeStep: dynamicParameters.timeStep,
                staticParameters: newtonStaticParameters,
                dynamicParamsT: dynamicParameters,
                dynamicParamsTplusDt: dynamicParameters,
                geometryT: geometry,
                geometryTplusDt: geometry,
                xOld: profiles.asTuple(
                    radialSpacing: mesh.radialSpacing,
                    boundaryConditions: dynamicParameters.boundaryConditions
                ),
                coreProfilesT: profiles,
                coreProfilesTplusDt: profiles,
                coeffsCallback: { currentProfiles, currentGeometry in
                    makeStableNewtonCoefficients(
                        for: currentProfiles,
                        geometry: currentGeometry
                    )
                }
            )
            evaluate(result.updatedProfiles)
        }
    }

    private func makeCoefficients(
        for currentProfiles: CoreProfiles,
        geometry currentGeometry: Geometry? = nil,
        sourceOverride: SourceTerms? = nil
    ) throws -> Block1DCoeffs {
        let activeGeometry = currentGeometry ?? geometry
        let factors = GeometricFactors.from(
            geometry: activeGeometry,
            evaluationMode: .deferred
        )
        let transport = transportModel.computeCoefficients(
            profiles: currentProfiles,
            geometry: activeGeometry,
            parameters: dynamicParameters.transportParameters
        )
        let activeSources: SourceTerms
        if let sourceOverride {
            activeSources = sourceOverride
        } else {
            activeSources = try sourceModel.computeTerms(
                in: SourceEvaluationContext(
                    profiles: currentProfiles,
                    geometry: activeGeometry,
                    geometricFactors: factors,
                    parameters: sourceParameters,
                    purpose: .solver
                )
            )
        }
        return buildBlock1DCoeffs(
            transport: transport,
            sources: activeSources,
            geometry: activeGeometry,
            staticParameters: staticParameters,
            profiles: currentProfiles,
            evaluationMode: .deferred,
            geometricFactors: factors
        )
    }

    private func makeStableNewtonCoefficients(
        for currentProfiles: CoreProfiles,
        geometry currentGeometry: Geometry
    ) -> Block1DCoeffs {
        let factors = GeometricFactors.from(
            geometry: currentGeometry,
            evaluationMode: .deferred
        )
        let transport = transportModel.computeCoefficients(
            profiles: currentProfiles,
            geometry: currentGeometry,
            parameters: dynamicParameters.transportParameters
        )
        let sources = SourceTerms.zero(
            cellCount: currentProfiles.ionTemperature.shape[0],
            evaluationMode: .deferred,
            metadata: nil,
            validateDebugUnits: false
        )
        return buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: currentGeometry,
            staticParameters: newtonStaticParameters,
            profiles: currentProfiles,
            evaluationMode: .deferred,
            geometricFactors: factors
        )
    }

    private func measure(
        name: String,
        operation: () throws -> Void
    ) throws -> RuntimeBenchmarkMetric {
        for _ in 0..<options.warmupIterations {
            try operation()
        }

        var samples: [Double] = []
        samples.reserveCapacity(options.measuredIterations)

        for _ in 0..<options.measuredIterations {
            let start = ProcessInfo.processInfo.systemUptime
            try operation()
            samples.append(ProcessInfo.processInfo.systemUptime - start)
        }

        return RuntimeBenchmarkMetric(
            name: name,
            cellCount: options.cellCount,
            warmupIterations: options.warmupIterations,
            measuredIterations: options.measuredIterations,
            totalSeconds: samples.reduce(0, +),
            meanMilliseconds: samples.mean * 1_000,
            medianMilliseconds: samples.median * 1_000,
            minimumMilliseconds: (samples.min() ?? 0) * 1_000,
            maximumMilliseconds: (samples.max() ?? 0) * 1_000
        )
    }

    private static func makeProfiles(cellCount: Int) -> CoreProfiles {
        let rho = MLXArray(0..<cellCount).asType(.float32) / Float(cellCount - 1)
        let ionTemperature = 100.0 + (12_000.0 - 100.0) * (1.0 - rho * rho)
        let electronTemperature = 100.0 + (10_000.0 - 100.0) * (1.0 - rho * rho)
        let electronDensity = 1e19 + (1e20 - 1e19) * (1.0 - 0.8 * rho * rho)
        let poloidalFlux = MLXArray.linspace(Float(0.0), Float(1.0), count: cellCount)

        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: ionTemperature),
            electronTemperature: EvaluatedArray(evaluating: electronTemperature),
            electronDensity: EvaluatedArray(evaluating: electronDensity),
            poloidalFlux: EvaluatedArray(evaluating: poloidalFlux)
        )
    }

    private static func makeDynamicParameters() throws -> DynamicRuntimeParameters {
        DynamicRuntimeParameters(
            timeStep: 1e-4,
            boundaryConditions: BoundaryConditions(
                ionTemperature: BoundaryCondition(left: .gradient(0.0), right: .value(100.0)),
                electronTemperature: BoundaryCondition(left: .gradient(0.0), right: .value(100.0)),
                electronDensity: BoundaryCondition(left: .gradient(0.0), right: .value(1e19)),
                poloidalFlux: BoundaryCondition(left: .value(0.0), right: .gradient(0.0))
            ),
            profileConditions: ProfileConditions(
                ionTemperature: .parabolic(peak: 12_000.0, edge: 100.0, exponent: 2.0),
                electronTemperature: .parabolic(peak: 10_000.0, edge: 100.0, exponent: 2.0),
                electronDensity: .parabolic(peak: 1e20, edge: 1e19, exponent: 2.0),
                currentDensity: .constant(0.0)
            ),
            sourceParameters: [:],
            transportParameters: try TransportParameters(
                modelType: .constant,
                parameters: [
                    "ionHeatDiffusivity": 1.0,
                    "electronHeatDiffusivity": 1.0,
                    "particleDiffusivity": 0.3,
                    "convectionVelocity": 0.0
                ]
            )
        )
    }

    private static func invalidCoefficients(
        cellCount: Int,
        geometry: Geometry,
        staticParameters: StaticRuntimeParameters,
        profiles: CoreProfiles
    ) -> Block1DCoeffs {
        let transport = TransportCoefficients(
            ionHeatDiffusivity: .full([cellCount], value: Float.nan),
            electronHeatDiffusivity: .full([cellCount], value: Float.nan),
            particleDiffusivity: .full([cellCount], value: Float.nan),
            convectionVelocity: .full([cellCount], value: Float.nan)
        )
        let sources = SourceTerms.invalidNumerics(cellCount: cellCount)
        return buildBlock1DCoeffs(
            transport: transport,
            sources: sources,
            geometry: geometry,
            staticParameters: staticParameters,
            profiles: profiles
        )
    }
}

private struct RuntimeBenchmarkReport: Codable {
    struct Configuration: Codable {
        let cellCount: Int
        let warmupIterations: Int
        let measuredIterations: Int
        let newtonIterations: Int
    }

    let generatedAt: String
    let configuration: Configuration
    let metrics: [RuntimeBenchmarkMetric]

    var summary: String {
        let rows = metrics.map { metric in
            let paddedName = metric.name.padding(toLength: 38, withPad: " ", startingAt: 0)
            return String(
                format: "%@ mean %8.3f ms  median %8.3f ms",
                paddedName,
                metric.meanMilliseconds,
                metric.medianMilliseconds
            )
        }
        return (["Runtime benchmarks"] + rows).joined(separator: "\n")
    }

    func write(to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: url)
    }
}

private struct RuntimeBenchmarkMetric: Codable {
    let name: String
    let cellCount: Int
    let warmupIterations: Int
    let measuredIterations: Int
    let totalSeconds: Double
    let meanMilliseconds: Double
    let medianMilliseconds: Double
    let minimumMilliseconds: Double
    let maximumMilliseconds: Double
}

private extension Array where Element == Double {
    var mean: Double {
        guard !isEmpty else {
            return 0
        }
        return reduce(0, +) / Double(count)
    }

    var median: Double {
        guard !isEmpty else {
            return 0
        }
        let sortedValues = sorted()
        let middle = sortedValues.count / 2
        if sortedValues.count.isMultiple(of: 2) {
            return (sortedValues[middle - 1] + sortedValues[middle]) / 2
        }
        return sortedValues[middle]
    }
}

private func evaluate(_ terms: SourceTerms) {
    eval(
        terms.ionHeating.value,
        terms.electronHeating.value,
        terms.particleSource.value,
        terms.currentSource.value
    )
}

private func evaluate(_ profiles: CoreProfiles) {
    eval(
        profiles.ionTemperature.value,
        profiles.electronTemperature.value,
        profiles.electronDensity.value,
        profiles.poloidalFlux.value
    )
}

private func evaluate(_ coeffs: Block1DCoeffs) {
    let equations = [
        coeffs.ionCoeffs,
        coeffs.electronCoeffs,
        coeffs.densityCoeffs,
        coeffs.fluxCoeffs
    ]

    for equation in equations {
        eval(
            equation.faceDiffusionCoefficient.value,
            equation.faceConvectionVelocity.value,
            equation.cellSource.value,
            equation.cellSourceMatrixCoefficient.value,
            equation.transientCoefficient.value
        )
    }

    eval(
        coeffs.geometry.cellVolumes.value,
        coeffs.geometry.faceAreas.value,
        coeffs.geometry.cellDistances.value,
        coeffs.geometry.cellRadii.value,
        coeffs.geometry.faceRadii.value,
        coeffs.geometry.jacobian.value,
        coeffs.geometry.majorRadiusMetric.value,
        coeffs.geometry.shapeMetric.value
    )
}
