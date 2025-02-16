#include <string>

#include "openmm/CalcDPDForceKernel.h"
#include "openmm/Platform.h"
#include "openmm/common/ComputeContext.h"

namespace OpenMM {

    class CommonCalcDPDForceKernel : public CalcDPDForceKernel {
    public:
        CommonCalcDPDForceKernel(std::string name, const Platform &platform,
                                 ComputeContext &cc, const System &system)
            : CalcDPDForceKernel(name, platform),
              cc(cc),
              hasInitializedKernel(false),
              system(system) {}

        void initialize(const System &system, const DPDForce &force) override;

        void copyParametersToContext(ContextImpl &context,
                                     const DPDForce &force) override;

        double execute(ContextImpl &context, bool includeForces,
                       bool includeEnergy, bool includeConservative) override;

    private:
        class ForceInfo;
        int numParticles, numTypes;
        bool hasInitializedKernel;
        ComputeContext &cc;
        ForceInfo *info;
        const System &system;
        ComputeArray particleTypeIndices, pairParams;
    };

}  // namespace OpenMM