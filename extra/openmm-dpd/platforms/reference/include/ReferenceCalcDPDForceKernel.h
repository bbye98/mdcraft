#ifndef OPENMM_REFERENCECALCDPDFORCEKERNEL_H_
#define OPENMM_REFERENCECALCDPDFORCEKERNEL_H_

#include "ReferenceNeighborList.h"
#include "ReferencePlatform.h"
#include "openmm/CalcDPDForceKernel.h"
#include "openmm/DPDForce.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/windowsExport.h"

namespace OpenMM {

    class ReferenceCalcDPDForceKernel : public CalcDPDForceKernel {
    public:
        ReferenceCalcDPDForceKernel(std::string name, const Platform &platform)
            : CalcDPDForceKernel(name, platform) {}

        ~ReferenceCalcDPDForceKernel();

        void initialize(const System &system, const DPDForce &force) override;

        double execute(ContextImpl &context, bool includeForces,
                       bool includeEnergy, bool includeConservative) override;

        void copyParametersToContext(ContextImpl &context,
                                     const DPDForce &force, int firstParticle,
                                     int lastParticle, int firstException,
                                     int lastException) override;

    private:
        DPDMethod dpdMethod;
        NeighborList *neighborList;
        int numParticles, numTypePairs, num14;
        bool exceptionsArePeriodic;
        double nonbondedCutoff;
        std::map<int, int> nb14Index;
        std::vector<int> particleTypes;
        std::vector<std::vector<int>> bonded14IndexArray;
        std::vector<std::vector<double>> particleParamArray, bonded14ParamArray;
        std::vector<std::array<double, 3>> baseExceptionParams;
        std::vector<std::set<int>> exclusions;

        void computeParameters(ContextImpl &context);
    };
}

#endif  // OPENMM_REFERENCECALCDPDFORCEKERNEL_H_