import Foundation
import Testing

@Suite("Repository quality gates")
struct RepositoryQualityGateTests {
    @Test("Source files do not expose deprecated API declarations")
    func sourceFilesDoNotExposeDeprecatedAPIs() throws {
        let violations = try sourceViolations(containing: "@available(*, deprecated")

        if !violations.isEmpty {
            Issue.record("Deprecated APIs must be removed instead of retained as compatibility aliases:\n\(violations.joined(separator: "\n"))")
        }

        #expect(violations.isEmpty)
    }

    @Test("Source files do not swallow errors with try question")
    func sourceFilesDoNotSwallowErrorsWithTryQuestion() throws {
        let violations = try sourceViolations(matching: "\\btry\\s*\\?")

        if !violations.isEmpty {
            Issue.record("Errors must be propagated or handled explicitly:\n\(violations.joined(separator: "\n"))")
        }

        #expect(violations.isEmpty)
    }

    @Test("Source files do not force unwrap throwing calls")
    func sourceFilesDoNotForceUnwrapThrowingCalls() throws {
        let violations = try sourceViolations(matching: "\\btry\\s*!")

        if !violations.isEmpty {
            Issue.record("Throwing errors must be handled explicitly instead of force-unwrapped:\n\(violations.joined(separator: "\n"))")
        }

        #expect(violations.isEmpty)
    }

    @Test("Source files do not contain obsolete short API names")
    func sourceFilesDoNotContainObsoleteShortAPINames() throws {
        let obsoleteNames = [
            "nCells",
            "initialDt",
            "minDt",
            "maxDt",
            "maxIterations",
            "chiIon",
            "chiElectron",
            "bohmFactor",
            "gyrobohmFactor",
            "defaultParameters",
            "withDefaults"
        ]

        let violations = try sourceViolations(containingAnyIdentifier: obsoleteNames)

        if !violations.isEmpty {
            Issue.record("Obsolete short names must not remain in source APIs:\n\(violations.joined(separator: "\n"))")
        }

        #expect(violations.isEmpty)
    }

    private func sourceViolations(containing needle: String) throws -> [String] {
        try swiftFiles(in: "Sources").flatMap { url in
            let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n", omittingEmptySubsequences: false)
            return lines.enumerated().compactMap { lineNumber, line -> String? in
                guard line.contains(needle) else {
                    return nil
                }

                return "\(relativePath(for: url)):\(lineNumber + 1): \(line.trimmingCharacters(in: .whitespaces))"
            }
        }
    }

    private func sourceViolations(matching pattern: String) throws -> [String] {
        let expression = try NSRegularExpression(pattern: pattern)

        return try swiftFiles(in: "Sources").flatMap { url in
            let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n", omittingEmptySubsequences: false)
            return lines.enumerated().compactMap { lineNumber, line -> String? in
                let lineString = String(line)
                let range = NSRange(lineString.startIndex..<lineString.endIndex, in: lineString)
                guard expression.firstMatch(in: lineString, range: range) != nil else {
                    return nil
                }

                return "\(relativePath(for: url)):\(lineNumber + 1): \(line.trimmingCharacters(in: .whitespaces))"
            }
        }
    }

    private func sourceViolations(containingAnyIdentifier names: [String]) throws -> [String] {
        let expressions = try names.map { name in
            (
                name,
                try NSRegularExpression(
                    pattern: "\\b\(NSRegularExpression.escapedPattern(for: name))\\b"
                )
            )
        }

        return try swiftFiles(in: "Sources").flatMap { url in
            let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n", omittingEmptySubsequences: false)
            return lines.enumerated().compactMap { lineNumber, line -> String? in
                let lineString = String(line)
                let range = NSRange(lineString.startIndex..<lineString.endIndex, in: lineString)
                guard let matchedName = expressions.first(where: { expression in
                    expression.1.firstMatch(in: lineString, range: range) != nil
                })?.0 else {
                    return nil
                }

                return "\(relativePath(for: url)):\(lineNumber + 1): \(matchedName): \(line.trimmingCharacters(in: .whitespaces))"
            }
        }
    }

    private func swiftFiles(in relativeDirectory: String) throws -> [URL] {
        let root = packageRoot.appendingPathComponent(relativeDirectory)
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var urls: [URL] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            if values.isRegularFile == true {
                urls.append(url)
            }
        }

        return urls.sorted { $0.path < $1.path }
    }

    private func relativePath(for url: URL) -> String {
        let rootPath = packageRoot.path + "/"
        return url.path.hasPrefix(rootPath) ? String(url.path.dropFirst(rootPath.count)) : url.path
    }

    private var packageRoot: URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 {
            url.deleteLastPathComponent()
        }
        return url
    }
}
