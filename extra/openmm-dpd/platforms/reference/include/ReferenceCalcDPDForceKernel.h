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
        bool exceptionsArePeriodic;
        int numParticles, numTypePairs;
        double nonbondedCutoff;
        std::vector<int> particleTypes;
        std::vector<std::set<int>> perParticleExclusions;
        std::vector<std::vector<double>> pairParams;

        void computeParameters(ContextImpl &context);
    };
}

#endif  // OPENMM_REFERENCECALCDPDFORCEKERNEL_H_