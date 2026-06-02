import Foundation

public struct ProfileValidationMatrix: Sendable {
    public let sourceName: String
    public let thresholds: ValidationThresholds
    public let results: [ComparisonResult]

    public init(
        sourceName: String,
        thresholds: ValidationThresholds,
        results: [ComparisonResult]
    ) {
        self.sourceName = sourceName
        self.thresholds = thresholds
        self.results = results
    }

    public var passed: Bool {
        results.allSatisfy(\.passed)
    }

    public var failedResults: [ComparisonResult] {
        results.filter { !$0.passed }
    }

    public var maximumL2Error: Float {
        results.map(\.l2Error).filter(\.isFinite).max() ?? Float.nan
    }

    public var maximumMAPE: Float {
        results.map(\.mape).filter(\.isFinite).max() ?? Float.nan
    }

    public var minimumCorrelation: Float {
        results.map(\.correlation).filter(\.isFinite).min() ?? Float.nan
    }

    public var failureSummary: String {
        guard !failedResults.isEmpty else {
            return "\(sourceName): all \(results.count) profile comparisons passed"
        }

        return failedResults
            .map { result in
                "\(result.quantity)@t=\(String(format: "%.6g", result.time)): "
                    + "L2=\(String(format: "%.3e", result.l2Error)) "
                    + "MAPE=\(String(format: "%.2f", result.mape))% "
                    + "corr=\(String(format: "%.4f", result.correlation))"
            }
            .joined(separator: "\n")
    }

    public static func compare(
        predicted: ReferenceProfiles,
        reference: ReferenceProfiles,
        sourceName: String = "reference",
        thresholds: ValidationThresholds = .torax,
        radiusTolerance: Float = 1e-5,
        timeTolerance: Float = 1e-6
    ) throws -> ProfileValidationMatrix {
        try validate(profiles: predicted, label: "predicted")
        try validate(profiles: reference, label: "reference")
        try validateComparableGrid(
            predicted: predicted,
            reference: reference,
            radiusTolerance: radiusTolerance,
            timeTolerance: timeTolerance
        )

        let results = [
            ProfileComparator.compare(
                quantity: "ion_temperature",
                predicted: predicted.ionTemperature,
                reference: reference.ionTemperature,
                time: reference.time,
                thresholds: thresholds
            ),
            ProfileComparator.compare(
                quantity: "electron_temperature",
                predicted: predicted.electronTemperature,
                reference: reference.electronTemperature,
                time: reference.time,
                thresholds: thresholds
            ),
            ProfileComparator.compare(
                quantity: "electron_density",
                predicted: predicted.electronDensity,
                reference: reference.electronDensity,
                time: reference.time,
                thresholds: thresholds
            )
        ]

        return ProfileValidationMatrix(
            sourceName: sourceName,
            thresholds: thresholds,
            results: results
        )
    }

    public static func compareTimeSeries(
        predicted: TORAXReferenceData,
        reference: TORAXReferenceData,
        sourceName: String = "torax",
        thresholds: ValidationThresholds = .torax,
        radiusTolerance: Float = 1e-5,
        timeTolerance: Float = 1e-6
    ) throws -> ProfileValidationMatrix {
        guard predicted.time.count == reference.time.count else {
            throw ProfileValidationMatrixError.timeCountMismatch(
                predicted: predicted.time.count,
                reference: reference.time.count
            )
        }

        var results: [ComparisonResult] = []
        for timeIndex in reference.time.indices {
            let snapshot = try compare(
                predicted: predicted.profiles(at: timeIndex),
                reference: reference.profiles(at: timeIndex),
                sourceName: sourceName,
                thresholds: thresholds,
                radiusTolerance: radiusTolerance,
                timeTolerance: timeTolerance
            )
            results.append(contentsOf: snapshot.results)
        }

        return ProfileValidationMatrix(
            sourceName: sourceName,
            thresholds: thresholds,
            results: results
        )
    }

    private static func validate(profiles: ReferenceProfiles, label: String) throws {
        let count = profiles.normalizedRadius.count
        guard count > 1 else {
            throw ProfileValidationMatrixError.emptyProfileSet(label: label)
        }

        try validate(values: profiles.normalizedRadius, label: label, quantity: "normalizedRadius", expectedCount: count)
        try validate(values: profiles.ionTemperature, label: label, quantity: "ionTemperature", expectedCount: count)
        try validate(values: profiles.electronTemperature, label: label, quantity: "electronTemperature", expectedCount: count)
        try validate(values: profiles.electronDensity, label: label, quantity: "electronDensity", expectedCount: count)

        for index in 1..<profiles.normalizedRadius.count {
            let previous = profiles.normalizedRadius[index - 1]
            let current = profiles.normalizedRadius[index]
            guard current > previous else {
                throw ProfileValidationMatrixError.nonMonotonicRadius(
                    label: label,
                    index: index,
                    previous: previous,
                    current: current
                )
            }
        }

        guard profiles.time.isFinite else {
            throw ProfileValidationMatrixError.nonFiniteValue(
                label: label,
                quantity: "time",
                index: 0,
                value: profiles.time
            )
        }
    }

    private static func validate(
        values: [Float],
        label: String,
        quantity: String,
        expectedCount: Int
    ) throws {
        guard values.count == expectedCount else {
            throw ProfileValidationMatrixError.inconsistentProfileLength(
                label: label,
                quantity: quantity,
                expected: expectedCount,
                actual: values.count
            )
        }

        for (index, value) in values.enumerated() where !value.isFinite {
            throw ProfileValidationMatrixError.nonFiniteValue(
                label: label,
                quantity: quantity,
                index: index,
                value: value
            )
        }
    }

    private static func validateComparableGrid(
        predicted: ReferenceProfiles,
        reference: ReferenceProfiles,
        radiusTolerance: Float,
        timeTolerance: Float
    ) throws {
        guard predicted.normalizedRadius.count == reference.normalizedRadius.count else {
            throw ProfileValidationMatrixError.inconsistentProfileLength(
                label: "predicted",
                quantity: "normalizedRadius",
                expected: reference.normalizedRadius.count,
                actual: predicted.normalizedRadius.count
            )
        }

        for index in reference.normalizedRadius.indices {
            let predictedRadius = predicted.normalizedRadius[index]
            let referenceRadius = reference.normalizedRadius[index]
            guard abs(predictedRadius - referenceRadius) <= radiusTolerance else {
                throw ProfileValidationMatrixError.radiusMismatch(
                    index: index,
                    predicted: predictedRadius,
                    reference: referenceRadius,
                    tolerance: radiusTolerance
                )
            }
        }

        guard abs(predicted.time - reference.time) <= timeTolerance else {
            throw ProfileValidationMatrixError.timeMismatch(
                predicted: predicted.time,
                reference: reference.time,
                tolerance: timeTolerance
            )
        }
    }
}
