#ifndef OPENMM_REFERENCECALCDPDFORCEKERNEL_H_
#define OPENMM_REFERENCECALCDPDFORCEKERNEL_H_

#include "openmm/CalcDPDForceKernel.h"
#include "openmm/DPDForce.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include "openmm/reference/ReferencePlatform.h"

namespace OpenMM {

    class ReferenceCalcDPDForceKernel : public CalcDPDForceKernel {
    public:
        ReferenceCalcDPDForceKernel(std::string name, const Platform &platform)
            : CalcDPDForceKernel(name, platform) {}

        ~ReferenceCalcDPDForceKernel();

        void initialize(const System &system, const DPDForce &force) override;

        void copyParametersToContext(ContextImpl &context,
                                     const DPDForce &force, int firstParticle,
                                     int lastParticle, int firstException,
                                     int lastException) override;

        double execute(ContextImpl &context, bool includeForces,
                       bool includeEnergy, bool includeConservative) override;

    private:
        DPDMethod dpdMethod;
        NeighborList *neighborList;
        bool exceptionsArePeriodic;
        int numParticles, numTypePairs, numExceptions, numTotalExceptions;
        double defaultA, defaultGamma, defaultRCut, temperature,
            nonbondedCutoff;
        std::vector<int> particleTypes;
        std::vector<std::array<int, 2>> exceptionParticlePairs;
        std::vector<std::array<double, 3>> pairParams, exceptionParams;
        std::vector<std::set<int>> perParticleExclusions;

        void calculateOneIxn(int ii, int jj,
                             const std::vector<OpenMM::Vec3> &positions,
                             const std::vector<OpenMM::Vec3> &velocities,
                             std::vector<OpenMM::Vec3> &forces,
                             double &totalEnergy, const double dt,
                             bool includeConservative, bool periodic,
                             const OpenMM::Vec3 *boxVectors = nullptr,
                             const std::array<double, 3> *params = nullptr);
    };
}

#endif  // OPENMM_REFERENCECALCDPDFORCEKERNEL_H_