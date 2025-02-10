#ifndef OPENMM_CALCDPDFORCEKERNEL_H_
#define OPENMM_CALCDPDFORCEKERNEL_H_

#include <iosfwd>
#include <set>
#include <string>
#include <vector>

#include "openmm/DPDForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/System.h"

namespace OpenMM {

    class CalcDPDForceKernel : public KernelImpl {
    public:
        enum DPDMethod {
            NoCutoff = 0,
            CutoffNonPeriodic = 1,
            CutoffPeriodic = 2
        };

        static std::string Name() { return "CalcDPDForce"; }

        CalcDPDForceKernel(std::string name, const Platform &platform)
            : KernelImpl(name, platform) {}

        virtual void initialize(const System &system,
                                const DPDForce &force) = 0;

        virtual double execute(ContextImpl &context, bool includeForces,
                               bool includeEnergy,
                               bool includeConservative) = 0;

        virtual void copyParametersToContext(
            ContextImpl &context, const DPDForce &force, int firstParticle,
            int lastParticle, int firstException, int lastException) = 0;
    };

}  // namespace OpenMM

#endif  // OPENMM_CALCDPDFORCEKERNEL_H_