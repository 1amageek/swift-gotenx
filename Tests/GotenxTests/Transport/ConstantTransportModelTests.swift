import Testing
import MLX
@testable import GotenxCore

@Suite("ConstantTransportModel Tests")
struct ConstantTransportModelTests {

    @Test("ConstantTransportModel initialization")
    func testInitialization() {
        let model = ConstantTransportModel(
            ionHeatDiffusivity: 1.0,
            electronHeatDiffusivity: 1.5,
            particleDiffusivity: 0.5,
            convectionVelocity: 0.0
        )

        #expect(model.name == "constant")
        #expect(model.ionHeatDiffusivityValue == 1.0)
        #expect(model.electronHeatDiffusivityValue == 1.5)
    }

    @Test("ConstantTransportModel initialization from parameters")
    func testInitializationFromParams() throws {
        let parameters = try TransportParameters(
            modelType: .constant,
            parameters: [
                "ionHeatDiffusivity": 2.0,
                "electronHeatDiffusivity": 2.5,
                "particleDiffusivity": 1.0,
                "convectionVelocity": 0.5
            ]
        )

        let model = try ConstantTransportModel(parameters: parameters)

        #expect(model.ionHeatDiffusivityValue == 2.0)
        #expect(model.electronHeatDiffusivityValue == 2.5)
        #expect(model.particleDiffusivityValue == 1.0)
        #expect(model.convectionVelocityValue == 0.5)
    }

    @Test("ConstantTransportModel computes uniform coefficients")
    func testComputeCoefficients() throws {
        let model = ConstantTransportModel(
            ionHeatDiffusivity: 1.0,
            electronHeatDiffusivity: 1.5
        )

        let profiles = CoreProfiles(
            ionTemperature: .full([10], value: Float(5.0)),
            electronTemperature: .full([10], value: Float(4.0)),
            electronDensity: .full([10], value: Float(3.0)),
            poloidalFlux: .full([10], value: Float(0.5))
        )

        let mesh = MeshConfig(
            cellCount: 10,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5
        )
        let geometry = createGeometry(from: mesh)

        let parameters = try TransportParameters(modelType: .constant)

        let coeffs = model.computeCoefficients(
            profiles: profiles,
            geometry: geometry,
            parameters: parameters
        )

        // Verify shape
        #expect(coeffs.ionHeatDiffusivity.shape == [10])
        #expect(coeffs.electronHeatDiffusivity.shape == [10])

        // Verify values are constant
        let chiIonArray = coeffs.ionHeatDiffusivity.value
        eval(chiIonArray)

        for i in 0..<10 {
            let value = chiIonArray[i].item(Float.self)
            #expect(abs(value - 1.0) < 1e-6)
        }
    }
}

@Suite("BohmGyroBohmTransportModel Tests")
struct BohmGyroBohmTransportModelTests {

    @Test("BohmGyroBohmTransportModel initialization")
    func testInitialization() {
        let model = BohmGyroBohmTransportModel(
            bohmCoefficient: 1.0,
            gyroBohmCoefficient: 1.0
        )

        #expect(model.name == "bohm-gyrobohm")
        #expect(model.bohmCoefficient == 1.0)
        #expect(model.gyroBohmCoefficient == 1.0)
    }

    @Test("BohmGyroBohmTransportModel computes diffusivities")
    func testComputeCoefficients() throws {
        let model = BohmGyroBohmTransportModel(
            bohmCoefficient: 1.0,
            gyroBohmCoefficient: 1.0
        )

        let profiles = CoreProfiles(
            ionTemperature: .full([10], value: Float(5.0)),
            electronTemperature: .full([10], value: Float(4.0)),
            electronDensity: .full([10], value: Float(3.0)),
            poloidalFlux: .full([10], value: Float(0.5))
        )

        let mesh = MeshConfig(
            cellCount: 10,
            majorRadius: 3.0,
            minorRadius: 1.0,
            toroidalField: 2.5
        )
        let geometry = createGeometry(from: mesh)

        let parameters = try TransportParameters(modelType: .bohmGyrobohm)

        let coeffs = model.computeCoefficients(
            profiles: profiles,
            geometry: geometry,
            parameters: parameters
        )

        // Verify shape
        #expect(coeffs.ionHeatDiffusivity.shape == [10])
        #expect(coeffs.electronHeatDiffusivity.shape == [10])

        // Verify values are positive
        let chiElectronArray = coeffs.electronHeatDiffusivity.value
        eval(chiElectronArray)

        for i in 0..<10 {
            let value = chiElectronArray[i].item(Float.self)
            #expect(value > 0.0)
        }
    }
}
