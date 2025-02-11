#ifndef OPENMM_CUDACALCDPDFORCEKERNEL_H_
#define OPENMM_CUDACALCDPDFORCEKERNEL_H_

#include "CudaContext.h"
#include "openmm/CalcDPDForceKernel.h"

namespace OpenMM {

    class CudaCalcDPDForceKernel : public CalcDPDForceKernel {
    public:
        CudaCalcDPDForceKernel(std::string name, const Platform& platform,
                               CudaContext& cu, const System& system)
            : CalcDPDForceKernel(name, platform), cu(cu) {}

        ~CudaCalcDPDForceKernel();

        void initialize(const System& system, const DPDForce& force) override;

        void copyParametersToContext(ContextImpl& context,
                                     const DPDForce& force, int firstParticle,
                                     int lastParticle, int firstException,
                                     int lastException) override;

        double execute(ContextImpl& context, bool includeForces,
                       bool includeEnergy, bool includeConservative) override;

    private:
        DPDMethod dpdMethod;
        CudaContext& cu;
    };
}

#endif  // OPENMM_CUDACALCDPDFORCEKERNEL_H_