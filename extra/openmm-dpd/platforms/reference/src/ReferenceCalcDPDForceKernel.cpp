#include "ReferenceCalcDPDForceKernel.h"

#include <array>
#include <vector>

OpenMM::ReferenceCalcDPDForceKernel::~ReferenceCalcDPDForceKernel() {
    if (neighborList != NULL)
        delete neighborList;
}

void OpenMM::ReferenceCalcDPDForceKernel::initialize(const System& system,
                                                     const DPDForce& force) {
    numParticles = force.getNumParticles();
    perParticleExclusions.resize(numParticles);
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double A, gamma, rCut;
        force.getExceptionParameters(i, particle1, particle2, A, gamma, rCut);
        perParticleExclusions[particle1].insert(particle2);
        perParticleExclusions[particle2].insert(particle1);
        // TODO: Handle exceptions (exclusions where A != 0).
    }

    numTypePairs = force.getNumTypePairs();
    pairParams.resize(numTypePairs * (numTypePairs + 1) / 2,
                      std::vector<double>(3));  // N * (N + 1) / 2
    for (int i = 0; i < numTypePairs; ++i) {
        int type1, type2;
        double A, gamma, rCut;
        force.getTypePairParameters(i, type1, type2, A, gamma, rCut);
        pairParams[type1 * (type1 + 1) / 2 + type2] = {
            A, gamma, rCut};  // i * (i + 1) / 2 + j
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