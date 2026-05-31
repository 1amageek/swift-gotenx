import Foundation
import Testing

func removeTestItemIfExists(at url: URL) {
    removeTestItemIfExists(atPath: url.path)
}

func removeTestItemIfExists(atPath path: String) {
    do {
        if FileManager.default.fileExists(atPath: path) {
            try FileManager.default.removeItem(atPath: path)
        }
    } catch {
        Issue.record("Failed to remove test item at \(path): \(error)")
    }
}
