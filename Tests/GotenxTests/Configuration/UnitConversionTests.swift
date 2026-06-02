// UnitConversionTests.swift
// Tests for unit conversion consistency

import Testing
import Foundation
@testable import GotenxCore

@Suite("Unit Conversion Tests")
struct UnitConversionTests {

    @Test("BoundaryConfig preserves eV units (no conversion)")
    func testBoundaryConfigTemperatureNoConversion() {
        let config = BoundaryConfig(
            ionTemperature: 1000.0,      // 1000 eV
            electronTemperature: 2000.0,  // 2000 eV
            electronDensity: 1e19,                // 10^19 m^-3
            type: .dirichlet
        )

        let bc = config.toBoundaryConditions()

        // Verify temperature: no conversion, stays in eV
        if case .value(let ti) = bc.ionTemperature.right {
            #expect(abs(ti - 1000.0) < 1e-6)  // 1000 eV (no conversion)
        } else {
            Issue.record("Expected .value constraint")
        }

        // Verify temperature: no conversion, stays in eV
        if case .value(let te) = bc.electronTemperature.right {
            #expect(abs(te - 2000.0) < 1e-6)  // 2000 eV (no conversion)
        } else {
            Issue.record("Expected .value constraint")
        }
    }

    @Test("BoundaryConfig preserves m^-3 units (no conversion)")
    func testBoundaryConfigDensityNoConversion() {
        let config = BoundaryConfig(
            ionTemperature: 100.0,
            electronTemperature: 100.0,
            electronDensity: 1e19,  // 10^19 m^-3
            type: .dirichlet
        )

        let bc = config.toBoundaryConditions()

        // Verify density: no conversion, stays in m^-3
        if case .value(let ne) = bc.electronDensity.right {
            #expect(abs(ne - 1e19) < 1e12)  // 1e19 m^-3 (no conversion)
        } else {
            Issue.record("Expected .value constraint")
        }
    }

    @Test("BoundaryConfig high density (no conversion)")
    func testBoundaryConfigHighDensity() {
        let config = BoundaryConfig(
            ionTemperature: 100.0,
            electronTemperature: 100.0,
            electronDensity: 5e20,  // 5 × 10^20 m^-3
            type: .dirichlet
        )

        let bc = config.toBoundaryConditions()

        // Verify: no conversion, stays in m^-3
        if case .value(let ne) = bc.electronDensity.right {
            #expect(abs(ne - 5e20) < 1e14)  // 5e20 m^-3 (no conversion)
        } else {
            Issue.record("Expected .value constraint")
        }
    }

    @Test("BoundaryConfig Neumann boundary (no value conversion)")
    func testBoundaryConfigNeumannBoundary() {
        let config = BoundaryConfig(
            ionTemperature: 1000.0,
            electronTemperature: 2000.0,
            electronDensity: 1e19,
            type: .neumann  // Gradient boundary
        )

        let bc = config.toBoundaryConditions()

        // Neumann boundaries should have gradient constraint
        if case .gradient(let grad) = bc.ionTemperature.right {
            #expect(grad == 0.0)  // Zero gradient
        } else {
            Issue.record("Expected .gradient constraint")
        }
    }

    @Test("ProfileConditions uses eV units (consistent with CoreProfiles)")
    func testProfileConditionsUnits() {
        let boundaries = BoundaryConfig(
            ionTemperature: 100.0,      // eV
            electronTemperature: 100.0,  // eV
            electronDensity: 1e19               // m^-3
        )

        let dynamicConfig = DynamicConfig(
            boundaries: boundaries,
            transport: .defaultConstant,
            initialProfile: .realistic  // Use 10× temperature ratio
        )

        let profileConditions = dynamicConfig.toProfileConditions()

        // ProfileConditions now uses eV (no conversion) for consistency with CoreProfiles
        if case .parabolic(let peak, let edge, _) = profileConditions.ionTemperature {
            // 100 eV edge (no conversion)
            #expect(abs(edge - 100.0) < 1e-6)
            // Core ~10× edge → 1000 eV
            #expect(abs(peak - 1000.0) < 1e-6)
        } else {
            Issue.record("Expected .parabolic profile")
        }
    }

    @Test("ProfileConditions uses m^-3 units (consistent with CoreProfiles)")
    func testProfileConditionsDensityUnits() {
        let boundaries = BoundaryConfig(
            ionTemperature: 100.0,
            electronTemperature: 100.0,
            electronDensity: 1e19  // m^-3
        )

        let dynamicConfig = DynamicConfig(
            boundaries: boundaries,
            transport: .defaultConstant,
            initialProfile: .realistic  // Use 3× density ratio
        )

        let profileConditions = dynamicConfig.toProfileConditions()

        // ProfileConditions now uses m^-3 (no conversion) for consistency with CoreProfiles
        if case .parabolic(let peak, let edge, _) = profileConditions.electronDensity {
            // 1e19 m^-3 edge (no conversion)
            #expect(abs(edge - 1e19) < 1e12)
            // Core ~3× edge → 3e19 m^-3
            #expect(abs(peak - 3e19) < 1e13)  // Larger tolerance for 3e19 magnitude
        } else {
            Issue.record("Expected .parabolic profile")
        }
    }

    @Test("DynamicRuntimeParameters uses eV, m^-3 (no conversion)")
    func testDynamicRuntimeParamsUnits() {
        let boundaries = BoundaryConfig(
            ionTemperature: 1000.0,  // 1000 eV
            electronTemperature: 2000.0,  // 2000 eV
            electronDensity: 1e20  // 1e20 m^-3
        )

        let dynamicConfig = DynamicConfig(
            boundaries: boundaries,
            transport: .defaultConstant
        )

        let runtimeParams = dynamicConfig.dynamicRuntimeParameters(timeStep: 0.01)

        // Verify boundary conditions use eV and m^-3 (no conversion)
        if case .value(let ti) = runtimeParams.boundaryConditions.ionTemperature.right {
            #expect(abs(ti - 1000.0) < 1e-6)  // 1000 eV (no conversion)
        }

        if case .value(let te) = runtimeParams.boundaryConditions.electronTemperature.right {
            #expect(abs(te - 2000.0) < 1e-6)  // 2000 eV (no conversion)
        }

        if case .value(let ne) = runtimeParams.boundaryConditions.electronDensity.right {
            #expect(abs(ne - 1e20) < 1e14)  // 1e20 m^-3 (no conversion)
        }
    }
}
