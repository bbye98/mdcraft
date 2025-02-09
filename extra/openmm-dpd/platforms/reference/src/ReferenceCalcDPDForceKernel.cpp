#include "ReferenceCalcDPDForceKernel.h"

#include <algorithm>
#include <array>
#include <vector>

OpenMM::ReferenceCalcDPDForceKernel::~ReferenceCalcDPDForceKernel() {
    if (neighborList != NULL)
        delete neighborList;
}

void OpenMM::ReferenceCalcDPDForceKernel::initialize(const System& system,
                                                     const DPDForce& force) {
    defaultA = force.getA();
    defaultGamma = force.getGamma();
    defaultRCut = force.getRCut();

    numParticles = force.getNumParticles();
    particleTypes.resize(numParticles);
    for (int i = 0; i < numParticles; ++i)
        particleTypes[i] = force.getParticleType(i);

    numTypePairs = force.getNumTypePairs();
    pairParams.resize(numTypePairs * (numTypePairs + 1) / 2);
    for (int i = 0; i < numTypePairs; ++i) {
        int type1, type2;
        double A, gamma, rCut;
        force.getTypePairParameters(i, type1, type2, A, gamma, rCut);
        pairParams[type1 * (type1 + 1) / 2 + type2] = {A, gamma, rCut};
    }

    std::vector<int> exceptionIndices;
    perParticleExclusions.resize(numParticles);
    for (int i = 0; i < force.getNumExceptions(); ++i) {
        int particle1, particle2;
        double A, gamma, rCut;
        force.getExceptionParameters(i, particle1, particle2, A, gamma, rCut);
        perParticleExclusions[particle1].insert(particle2);
        perParticleExclusions[particle2].insert(particle1);
        if (A != 0)
            exceptionIndices.push_back(i);
    }

    numExceptions = exceptionIndices.size();
    numTotalExceptions = force.getNumExceptions();
    exceptionParticlePairs.resize(numExceptions);
    exceptionParams.resize(numExceptions);
    for (int i = 0; i < numExceptions; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(
            exceptionIndices[i], exceptionParticlePairs[i][0],
            exceptionParticlePairs[i][1], exceptionParams[i][0],
            exceptionParams[i][1], exceptionParams[i][2]);
    }

    dpdMethod = CalcDPDForceKernel::DPDMethod(force.getDPDMethod());
    nonbondedCutoff = force.getCutoffDistance();
    neighborList = new NeighborList();
    if (dpdMethod == CutoffNonPeriodic)
        exceptionsArePeriodic = false;
    else
        exceptionsArePeriodic =
            force.getExceptionsUsePeriodicBoundaryConditions();
}

void OpenMM::ReferenceCalcDPDForceKernel::copyParametersToContext(
    ContextImpl& context, const DPDForce& force, int firstParticle,
    int lastParticle, int firstException, int lastException) {
    if (force.getNumParticles() != numParticles)
        throw OpenMMException(
            "DPDForce.updateParametersInContext: The number of particles has "
            "changed");
    if (force.getNumTypePairs() != numTypePairs)
        throw OpenMMException(
            "DPDForce.updateParametersInContext: The number of type pairs has "
            "changed");
    if (force.getNumExceptions() != numTotalExceptions)
        throw OpenMMException(
            "DPDForce.updateParametersInContext: The number of exceptions has "
            "changed");

    defaultA = force.getA();
    defaultGamma = force.getGamma();
    defaultRCut = force.getRCut();

    std::unordered_set<int> uniqueTypesSet;
    for (int i = 0; i < numParticles; i++) {
        particleTypes[i] = force.getParticleType(i);
        if (particleTypes[i] != 0)
            uniqueTypesSet.insert(particleTypes[i]);
    }
    std::vector<int> uniqueTypesVector(uniqueTypesSet.begin(),
                                       uniqueTypesSet.end());
    std::sort(uniqueTypesVector.begin(), uniqueTypesVector.end());
    for (int i = 0; i < uniqueTypesVector.size(); ++i) {
        int type1 = uniqueTypesVector[i];
        if (type1 == 0)
            continue;
        for (int j = i; j < uniqueTypesVector.size(); ++j) {
            int type2 = uniqueTypesVector[j];
            if (type2 == 0)
                continue;
            if (force.getTypePairIndex(type1, type2) == -1) {
                throw OpenMM::OpenMMException(
                    "DPDForce: No DPD parameters defined for particles of "
                    "types " +
                    std::to_string(type1) + " and " + std::to_string(type2));
            }
        }
    }

    for (int i = 0; i < numTypePairs; ++i) {
        int type1, type2;
        double A, gamma, rCut;
        force.getTypePairParameters(i, type1, type2, A, gamma, rCut);
        pairParams[type1 * (type1 + 1) / 2 + type2] = {A, gamma, rCut};
    }

    std::vector<int> exceptionIndices;
    for (int i = 0; i < force.getNumExceptions(); ++i) {
        int particle1, particle2;
        double A, gamma, rCut;
        force.getExceptionParameters(i, particle1, particle2, A, gamma, rCut);
        if (A != 0)
            exceptionIndices.push_back(i);
    }
    if (exceptionIndices.size() != numExceptions)
        throw OpenMMException(
            "DPDForce.updateParametersInContext: The number of non-excluded "
            "exceptions has changed");

    for (int i = 0; i < numExceptions; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(
            exceptionIndices[i], exceptionParticlePairs[i][0],
            exceptionParticlePairs[i][1], exceptionParams[i][0],
            exceptionParams[i][1], exceptionParams[i][2]);
    }
}

double OpenMM::ReferenceCalcDPDForceKernel::execute(ContextImpl& context,
                                                    bool includeForces,
                                                    bool includeEnergy,
                                                    bool includeConservative) {
    ReferencePlatform::PlatformData* data =
        reinterpret_cast<ReferencePlatform::PlatformData*>(
            context.getPlatformData());
    std::vector<Vec3>& particlePositions = *data->positions;
    std::vector<Vec3>& particleForces = *data->forces;
}