#include <algorithm>
#include <iostream>
#include <set>

#include "CudaBondedUtilities.h"
#include "CudaForceInfo.h"
#include "CudaICKernelSources.h"
#include "CudaICKernels.h"
#include "CudaIntegrationUtilities.h"
#include "ICLangevinIntegrator.h"
#include "SimTKOpenMMRealType.h"
#include "openmm/CMMotionRemover.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace OpenMM;
using namespace std;

CudaIntegrateICLangevinStepKernel::~CudaIntegrateICLangevinStepKernel() {
    cu.setAsCurrent();
    if (params != NULL) delete params;
    if (invAtomIndex != NULL) delete invAtomIndex;
}

void CudaIntegrateICLangevinStepKernel::initialize(
    const System& system, const ICPlugin::ICLangevinIntegrator& integrator) {
    cu.getPlatformData().initializeContexts(system);
    cu.setAsCurrent();
    cu.getIntegrationUtilities().initRandomNumberGenerator(
        integrator.getRandomNumberSeed());
    map<string, string> defines;
    CUmodule module = cu.createModule(
        CudaICKernelSources::vectorOps + CudaICKernelSources::ICLangevin,
        defines, "");
    kernel1 = cu.getKernel(module, "integrateICLangevinPart1");
    kernel2 = cu.getKernel(module, "integrateICLangevinPart2");
    kernelImage = cu.getKernel(module, "updateImageParticlePositions");
    kernelReorder = cu.getKernel(module, "reorderInverseAtomOrderIndices");
    params = new CudaArray(
        cu, 3,
        cu.getUseDoublePrecision() || cu.getUseMixedPrecision() ? sizeof(double)
                                                                : sizeof(float),
        "langevinParams");
    prevStepSize = -1.0;

    // Check image particles are properly set (same number as original, mass =
    // 0).

    int numCells = integrator.getNumCells();
    if (numCells % 2 != 0)
        throw OpenMMException("Number of cells must be even");
    int numAtoms = system.getNumParticles();
    if (numAtoms % numCells != 0)
        throw OpenMMException(
            "Number of particles is not a multiple of the number of cells");
    for (int i = numAtoms / numCells; i < numAtoms; i++) {
        if (system.getParticleMass(i) != 0.0)
            throw OpenMMException("Image particle has nonzero mass");
    }

    // Check the unit cell z-dimension.

    Vec3 a, b, c;
    system.getDefaultPeriodicBoxVectors(a, b, c);
    cellZSize = integrator.getCellZSize();
    if (cellZSize < 0) {
        cellZSize = c[2] / numCells;
    } else if (cellZSize * numCells != c[2]) {
        throw OpenMMException(
            "Unit cell dimension does not match the provided cellZSize value");
    }

    // Initialize the positions of the image particles.

    invAtomIndex =
        CudaArray::create<int>(cu, cu.getPaddedNumAtoms(), "invAtomIndex");
}

void CudaIntegrateICLangevinStepKernel::execute(
    ContextImpl& context, const ICPlugin::ICLangevinIntegrator& integrator) {
    cu.setAsCurrent();
    CudaIntegrationUtilities& integration = cu.getIntegrationUtilities();
    int numAtoms = cu.getNumAtoms();
    int numCells = integrator.getNumCells();
    int numRealAtoms = numAtoms / numCells;
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    double temperature = integrator.getTemperature();
    double friction = integrator.getFriction();
    double stepSize = integrator.getStepSize();
    cu.getIntegrationUtilities().setNextStepSize(stepSize);
    if (temperature != prevTemp || friction != prevFriction ||
        stepSize != prevStepSize) {
        // Recreate the computation objects with the new parameters.

        double kT = BOLTZ * temperature;
        double vscale = exp(-stepSize * friction);
        double fscale = (friction == 0 ? stepSize : (1 - vscale) / friction);
        double noisescale = sqrt(kT * (1 - vscale * vscale));
        if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
            vector<double> p(params->getSize());
            p[0] = vscale;
            p[1] = fscale;
            p[2] = noisescale;
            params->upload(p);
        } else {
            vector<float> p(params->getSize());
            p[0] = (float)vscale;
            p[1] = (float)fscale;
            p[2] = (float)noisescale;
            params->upload(p);
        }
        prevTemp = temperature;
        prevFriction = friction;
        prevStepSize = stepSize;
    }

    if (cu.getAtomsWereReordered()) {
        void* argsReorder[] = {&numAtoms,
                               &cu.getAtomIndexArray().getDevicePointer(),
                               &invAtomIndex->getDevicePointer()};
        cu.executeKernel(kernelReorder, argsReorder, numAtoms, 128);
    }

    // Call the first integration kernel.

    int randomIndex = integration.prepareRandomNumbers(cu.getPaddedNumAtoms());
    void* args1[] = {&numAtoms,
                     &paddedNumAtoms,
                     &cu.getVelm().getDevicePointer(),
                     &cu.getForce().getDevicePointer(),
                     &integration.getPosDelta().getDevicePointer(),
                     &params->getDevicePointer(),
                     &integration.getStepSize().getDevicePointer(),
                     &integration.getRandom().getDevicePointer(),
                     &randomIndex};
    cu.executeKernel(kernel1, args1, numAtoms, 128);

    // Apply constraints.

    integration.applyConstraints(integrator.getConstraintTolerance());

    // Call the second integration kernel.

    CUdeviceptr posCorrection =
        (cu.getUseMixedPrecision() ? cu.getPosqCorrection().getDevicePointer()
                                   : 0);
    void* args2[] = {&numAtoms,
                     &cu.getPosq().getDevicePointer(),
                     &integration.getPosDelta().getDevicePointer(),
                     &cu.getVelm().getDevicePointer(),
                     &integration.getStepSize().getDevicePointer(),
                     &posCorrection};
    cu.executeKernel(kernel2, args2, numAtoms, 128);
    integration.computeVirtualSites();

    // Call the image charge position update kernel.

    void* argsImage[] = {&numRealAtoms,  &numCells,
                         &cellZSize,     &cu.getPosq().getDevicePointer(),
                         &posCorrection, &invAtomIndex->getDevicePointer()};
    cu.executeKernel(kernelImage, argsImage, numRealAtoms, 128);

    // Update the time and step count.

    cu.setTime(cu.getTime() + stepSize);
    cu.setStepCount(cu.getStepCount() + 1);
    cu.reorderAtoms();
}

double CudaIntegrateICLangevinStepKernel::computeKineticEnergy(
    ContextImpl& context, const ICPlugin::ICLangevinIntegrator& integrator) {
    return cu.getIntegrationUtilities().computeKineticEnergy(
        0.5 * integrator.getStepSize());
}

void CudaIntegrateICDrudeLangevinStepKernel::initialize(
    const System& system, const ICPlugin::ICDrudeLangevinIntegrator& integrator,
    const DrudeForce& force) {
    cu.getPlatformData().initializeContexts(system);
    cu.getIntegrationUtilities().initRandomNumberGenerator(
        (unsigned int)integrator.getRandomNumberSeed());

    // Check image particles are properly set (same number as original, mass =
    // 0).

    int numCells = integrator.getNumCells();
    if (numCells % 2 != 0)
        throw OpenMMException("Number of cells must be even");
    int numAtoms = system.getNumParticles();
    if (numAtoms % numCells != 0)
        throw OpenMMException(
            "Number of particles is not a multiple of the number of cells");
    for (int i = numAtoms / numCells; i < numAtoms; i++) {
        if (system.getParticleMass(i) != 0.0)
            throw OpenMMException("Image particle has nonzero mass");
    }
    int numRealAtoms = numAtoms / numCells;
    invAtomIndex.initialize<int>(cu, cu.getPaddedNumAtoms(), "invAtomIndex");

    // Check the unit cell z-dimension.

    Vec3 a, b, c;
    system.getDefaultPeriodicBoxVectors(a, b, c);
    cellZSize = integrator.getCellZSize();
    if (cellZSize < 0) {
        cellZSize = c[2] / numCells;
    } else if (cellZSize * numCells != c[2]) {
        throw OpenMMException(
            "Unit cell dimension does not match the provided cellZSize value");
    }

    // Identify particle pairs and ordinary particles.

    set<int> particles;
    vector<int> normalParticleVec;
    vector<int2> pairParticleVec;
    for (int i = 0; i < numRealAtoms; i++) particles.insert(i);
    for (int i = 0; i < force.getNumParticles(); i++) {
        int p, p1, p2, p3, p4;
        double charge, polarizability, aniso12, aniso34;
        force.getParticleParameters(i, p, p1, p2, p3, p4, charge,
                                    polarizability, aniso12, aniso34);
        particles.erase(p);
        particles.erase(p1);
        pairParticleVec.push_back(make_int2(p, p1));
    }
    normalParticleVec.insert(normalParticleVec.begin(), particles.begin(),
                             particles.end());
    normalParticles.initialize<int>(cu, max((int)normalParticleVec.size(), 1),
                                    "drudeNormalParticles");
    pairParticles.initialize<int2>(cu, max((int)pairParticleVec.size(), 1),
                                   "drudePairParticles");
    if (normalParticleVec.size() > 0) normalParticles.upload(normalParticleVec);
    if (pairParticleVec.size() > 0) pairParticles.upload(pairParticleVec);

    // Create kernels.

    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(numRealAtoms);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_NORMAL_PARTICLES"] = cu.intToString(normalParticleVec.size());
    defines["NUM_PAIRS"] = cu.intToString(pairParticleVec.size());
    map<string, string> replacements;
    CUmodule module = cu.createModule(CudaICKernelSources::vectorOps +
                                          CudaICKernelSources::ICDrudeLangevin +
                                          CudaICKernelSources::ICLangevin,
                                      defines, "");
    kernel1 = cu.getKernel(module, "integrateICDrudeLangevinPart1");
    kernel2 = cu.getKernel(module, "integrateICDrudeLangevinPart2");
    hardwallKernel = cu.getKernel(module, "applyHardWallConstraints");
    kernelImage = cu.getKernel(module, "updateImageParticlePositions");
    kernelReorder = cu.getKernel(module, "reorderInverseAtomOrderIndexes");
    prevStepSize = -1.0;
}

void CudaIntegrateICDrudeLangevinStepKernel::execute(
    ContextImpl& context,
    const ICPlugin::ICDrudeLangevinIntegrator& integrator) {
    cu.setAsCurrent();
    CudaIntegrationUtilities& integration = cu.getIntegrationUtilities();
    int numAtoms = cu.getNumAtoms();
    int numCells = integrator.getNumCells();
    int numRealAtoms = numAtoms / numCells;

    // Compute integrator coefficients.

    double stepSize = integrator.getStepSize();
    double vscale = exp(-stepSize * integrator.getFriction());
    double fscale =
        (1 - vscale) / integrator.getFriction() / (double)0x100000000;
    double noisescale =
        sqrt(2 * BOLTZ * integrator.getTemperature() *
             integrator.getFriction()) *
        sqrt(0.5 * (1 - vscale * vscale) / integrator.getFriction());
    double vscaleDrude = exp(-stepSize * integrator.getDrudeFriction());
    double fscaleDrude =
        (1 - vscaleDrude) / integrator.getDrudeFriction() / (double)0x100000000;
    double noisescaleDrude = sqrt(2 * BOLTZ * integrator.getDrudeTemperature() *
                                  integrator.getDrudeFriction()) *
                             sqrt(0.5 * (1 - vscaleDrude * vscaleDrude) /
                                  integrator.getDrudeFriction());
    double maxDrudeDistance = integrator.getMaxDrudeDistance();
    double hardwallscaleDrude = sqrt(BOLTZ * integrator.getDrudeTemperature());
    if (stepSize != prevStepSize) {
        if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
            double2 ss = make_double2(0, stepSize);
            integration.getStepSize().upload(&ss);
        } else {
            float2 ss = make_float2(0, (float)stepSize);
            integration.getStepSize().upload(&ss);
        }
        prevStepSize = stepSize;
    }

    // Create appropriate pointer for the precision mode.

    float vscaleFloat = (float)vscale;
    float fscaleFloat = (float)fscale;
    float noisescaleFloat = (float)noisescale;
    float vscaleDrudeFloat = (float)vscaleDrude;
    float fscaleDrudeFloat = (float)fscaleDrude;
    float noisescaleDrudeFloat = (float)noisescaleDrude;
    float maxDrudeDistanceFloat = (float)maxDrudeDistance;
    float hardwallscaleDrudeFloat = (float)hardwallscaleDrude;
    void *vscalePtr, *fscalePtr, *noisescalePtr, *vscaleDrudePtr,
        *fscaleDrudePtr, *noisescaleDrudePtr, *maxDrudeDistancePtr,
        *hardwallscaleDrudePtr;
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        vscalePtr = &vscale;
        fscalePtr = &fscale;
        noisescalePtr = &noisescale;
        vscaleDrudePtr = &vscaleDrude;
        fscaleDrudePtr = &fscaleDrude;
        noisescaleDrudePtr = &noisescaleDrude;
        maxDrudeDistancePtr = &maxDrudeDistance;
        hardwallscaleDrudePtr = &hardwallscaleDrude;
    } else {
        vscalePtr = &vscaleFloat;
        fscalePtr = &fscaleFloat;
        noisescalePtr = &noisescaleFloat;
        vscaleDrudePtr = &vscaleDrudeFloat;
        fscaleDrudePtr = &fscaleDrudeFloat;
        noisescaleDrudePtr = &noisescaleDrudeFloat;
        maxDrudeDistancePtr = &maxDrudeDistanceFloat;
        hardwallscaleDrudePtr = &hardwallscaleDrudeFloat;
    }

    if (cu.getAtomsWereReordered()) {
        void* argsReorder[] = {&numAtoms,
                               &cu.getAtomIndexArray().getDevicePointer(),
                               &invAtomIndex.getDevicePointer()};
        cu.executeKernel(kernelReorder, argsReorder, numAtoms, 128);
    }

    // Call the first integration kernel.

    int randomIndex = integration.prepareRandomNumbers(
        normalParticles.getSize() + 2 * pairParticles.getSize());
    void* args1[] = {&cu.getVelm().getDevicePointer(),
                     &cu.getForce().getDevicePointer(),
                     &integration.getPosDelta().getDevicePointer(),
                     &normalParticles.getDevicePointer(),
                     &pairParticles.getDevicePointer(),
                     &integration.getStepSize().getDevicePointer(),
                     vscalePtr,
                     fscalePtr,
                     noisescalePtr,
                     vscaleDrudePtr,
                     fscaleDrudePtr,
                     noisescaleDrudePtr,
                     &integration.getRandom().getDevicePointer(),
                     &randomIndex};
    cu.executeKernel(kernel1, args1, numRealAtoms);

    // Apply constraints.

    integration.applyConstraints(integrator.getConstraintTolerance());

    // Call the second integration kernel.

    CUdeviceptr posCorrection =
        (cu.getUseMixedPrecision() ? cu.getPosqCorrection().getDevicePointer()
                                   : 0);
    void* args2[] = {&cu.getPosq().getDevicePointer(),
                     &integration.getPosDelta().getDevicePointer(),
                     &cu.getVelm().getDevicePointer(),
                     &integration.getStepSize().getDevicePointer(),
                     &posCorrection};
    cu.executeKernel(kernel2, args2, numRealAtoms);

    // Apply hard wall constraints.

    if (maxDrudeDistance > 0) {
        void* hardwallArgs[] = {&cu.getPosq().getDevicePointer(),
                                &posCorrection,
                                &cu.getVelm().getDevicePointer(),
                                &pairParticles.getDevicePointer(),
                                &integration.getStepSize().getDevicePointer(),
                                maxDrudeDistancePtr,
                                hardwallscaleDrudePtr};
        cu.executeKernel(hardwallKernel, hardwallArgs, pairParticles.getSize());
    }
    integration.computeVirtualSites();

    // Call the image charge position update kernel.

    void* argsImage[] = {&numRealAtoms,  &numCells,
                         &cellZSize,     &cu.getPosq().getDevicePointer(),
                         &posCorrection, &invAtomIndex.getDevicePointer()};
    cu.executeKernel(kernelImage, argsImage, numRealAtoms, 128);

    // Update the time and step count.

    cu.setTime(cu.getTime() + stepSize);
    cu.setStepCount(cu.getStepCount() + 1);
    cu.reorderAtoms();
}

double CudaIntegrateICDrudeLangevinStepKernel::computeKineticEnergy(
    ContextImpl& context,
    const ICPlugin::ICDrudeLangevinIntegrator& integrator) {
    return cu.getIntegrationUtilities().computeKineticEnergy(
        0.5 * integrator.getStepSize());
}