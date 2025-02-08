#include "ReferenceCalcDPDForceKernel.h"

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

    std::vector<int> exceptionIndices;
    numParticles = force.getNumParticles();
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

    numTypePairs = force.getNumTypePairs();
    numExceptions = exceptionIndices.size();
    pairParams.resize(numTypePairs * (numTypePairs + 1) / 2);
    exceptionParticlePairs.resize(numExceptions);
    exceptionParams.resize(numExceptions);
    for (int i = 0; i < numTypePairs; ++i) {
        int type1, type2;
        double A, gamma, rCut;
        force.getTypePairParameters(i, type1, type2, A, gamma, rCut);
        pairParams[type1 * (type1 + 1) / 2 + type2] = {A, gamma, rCut};
    }
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

double OpenMM::ReferenceCalcDPDForceKernel::execute(ContextImpl& context,
                                                    bool includeForces,
                                                    bool includeEnergy,
                                                    bool includeConservative) {}

void OpenMM::ReferenceCalcDPDForceKernel::copyParametersToContext(
    ContextImpl& context, const DPDForce& force, int firstParticle,
    int lastParticle, int firstException, int lastException) {}

void OpenMM::ReferenceCalcDPDForceKernel::computeParameters(
    ContextImpl& context) {}