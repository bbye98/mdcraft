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
}

void OpenMM::CudaCalcDPDForceKernel::copyParametersToContext(
    ContextImpl& context, const DPDForce& force, int firstParticle,
    int lastParticle, int firstException, int lastException) {}

double OpenMM::CudaCalcDPDForceKernel::execute(ContextImpl& context,
                                               bool includeForces,
                                               bool includeEnergy,
                                               bool includeConservative) {}