#ifndef OPENMM_REFERENCECALCDPDFORCEKERNEL_H_
#define OPENMM_REFERENCECALCDPDFORCEKERNEL_H_

#include <array>
#include <map>
#include <string>
#include <vector>

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

        void initialize(const System &system, const DPDForce &force) override;

        void copyParametersToContext(ContextImpl &context,
                                     const DPDForce &force) override;

        double execute(ContextImpl &context, bool includeForces,
                       bool includeEnergy, bool includeConservative) override;

    private:
        CalcDPDForceKernel::NonbondedMethod nonbondedMethod;
        NeighborList *neighborList;
        bool exceptionsArePeriodic;
        int numParticles, numTypes, numExceptions, numTotalExceptions;
        double defaultA, defaultGamma, defaultRCut, temperature,
            nonbondedCutoff;
        std::map<int, int> typeIndexMap;
        std::vector<int> particleTypes;
        std::vector<std::array<int, 2>> exceptionParticlePairs;
        std::vector<std::array<double, 3>> pairParams, exceptionParams;
        std::vector<std::set<int>> perParticleExclusions;

        void calculateOneIxn(int ii, int jj, const std::vector<Vec3> &positions,
                             const std::vector<Vec3> &velocities,
                             std::vector<Vec3> &forces, double &totalEnergy,
                             const double dt, bool includeConservative,
                             bool periodic, const Vec3 *boxVectors = nullptr,
                             const std::array<double, 3> *params = nullptr);
    };
}

#endif  // OPENMM_REFERENCECALCDPDFORCEKERNEL_H_