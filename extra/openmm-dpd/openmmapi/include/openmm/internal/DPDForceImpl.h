#ifndef OPENMM_DPDFORCEIMPL_H_
#define OPENMM_DPDFORCEIMPL_H_

#include <string>
#include <utility>
#include <vector>

#include "ForceImpl.h"
#include "openmm/DPDForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/windowsExport.h"

namespace OpenMM {

    class System;

    class OPENMM_EXPORT DPDForceImpl : public ForceImpl {
    public:
        DPDForceImpl(const DPDForce &owner);

        ~DPDForceImpl();

        const DPDForce &getOwner() const { return owner; }

        void initialize(ContextImpl &context);

        std::vector<std::string> getKernelNames() {
            return {OpenMM::CalcDPDForceKernel::Name()};
        }

        std::map<std::string, double> getDefaultParameters();

        void updateParametersInContext(ContextImpl &context, int firstParticle,
                                       int lastParticle, int firstException,
                                       int lastException);

        void updateContextState(ContextImpl &context, bool &forcesInvalid) {}

        double calcForcesAndEnergy(ContextImpl &context, bool includeForces,
                                   bool includeEnergy, int groups);

    private:
        const DPDForce &owner;
        Kernel kernel;
        bool includeConservative;
    };

}  // namespace OpenMM

#endif  // OPENMM_DPDFORCEIMPL_H_