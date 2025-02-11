#include "CudaCalcDPDForceKernel.h"

#include "openmm/common/ContextSelector.h"

void OpenMM::CudaCalcDPDForceKernel::initialize(const System& system,
                                                const DPDForce& force) {
    ContextSelector selector(cu);
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() &&
                         &system.getForce(forceIndex) != &force;
         ++forceIndex);
    std::string prefix = "dpd" + cu.intToString(forceIndex) + "_";

    std::vector<std::pair<int, int>> excludedPairs;
    std::vector<int> exceptionIndices;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double A, gamma, rCut;
        force.getExceptionParameters(i, particle1, particle2, A, gamma, rCut);
        excludedPairs.emplace_back(std::make_pair(particle1, particle2));
        if (A != 0) {
            exceptionIndices.push_back(i);
        }
    }

    int numParticles = force.getNumParticles();
    // TODO
}

void OpenMM::CudaCalcDPDForceKernel::copyParametersToContext(
    ContextImpl& context, const DPDForce& force, int firstParticle,
    int lastParticle, int firstException, int lastException) {}

double OpenMM::CudaCalcDPDForceKernel::execute(ContextImpl& context,
                                               bool includeForces,
                                               bool includeEnergy,
                                               bool includeConservative) {}