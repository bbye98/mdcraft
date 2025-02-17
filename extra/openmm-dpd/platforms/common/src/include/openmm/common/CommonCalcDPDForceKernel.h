#include <string>
#include <tuple>

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
        class ReorderListener;
        ComputeContext &cc;
        ForceInfo *info;
        const System &system;
        DPDForce::NonbondedMethod nonbondedMethod;
        bool hasInitializedKernel;
        int numParticles, maxNeighborBlocks;
        ComputeArray particleTypeIndices, pairParams, sortedParticles;
        ComputeArray exclusions, exclusionStartIndex, exceptionParticles,
            exceptionParams;
        ComputeArray blockCenter, blockBoundingBox, sortedPositions, neighbors,
            neighborIndex, neighborBlockCount;
        ComputeEvent event;
        // ComputeKernel framesKernel, blockBoundsKernel, neighborsKernel,
        // forceKernel;
        std::vector<std::pair<int, int>> exceptionPairs, excludedPairs;

        void sortAtoms();
    };

}  // namespace OpenMM