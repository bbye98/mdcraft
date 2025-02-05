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
    exclusions.resize(numParticles);
    std::vector<int> nb14s;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double A, gamma, rCut;
        force.getExceptionParameters(i, particle1, particle2, A, gamma, rCut);
        exclusions[particle1].insert(particle2);
        exclusions[particle2].insert(particle1);
        if (A != 0.0) {
            nb14Index[i] = nb14s.size();
            nb14s.push_back(i);
        }
    }

    numTypePairs = force.getNumTypePairs();
    particleParamArray.resize(numParticles, std::vector<double>(3));
    for (int i = 0; i < numTypePairs; ++i) {
        int type1, type2;
        double A, gamma, rCut;
        force.getTypePairParameters(i, type1, type2, A, gamma, rCut);
        particleParamArray[type1 * (type1 + 1) / 2 + type2] = {A, gamma, rCut};
    }

    num14 = nb14s.size();
    bonded14IndexArray.resize(num14, std::vector<int>(2));
    bonded14ParamArray.resize(num14, std::vector<double>(3));
    baseExceptionParams.resize(num14, std::array<double, 3>());
    for (int i = 0; i < num14; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(
            nb14s[i], particle1, particle2, baseExceptionParams[i][0],
            baseExceptionParams[i][1], baseExceptionParams[i][2]);
        bonded14IndexArray[i][0] = particle1;
        bonded14IndexArray[i][1] = particle2;
    }
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