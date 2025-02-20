#include "openmm/common/CommonCalcDPDForceKernel.h"

#include <map>

#include "CommonKernelSources.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ComputeParameterSet.h"
#include "openmm/common/ContextSelector.h"

class OpenMM::CommonCalcDPDForceKernel::ForceInfo
    : public OpenMM::ComputeForceInfo {
public:
    ForceInfo(const OpenMM::DPDForce &force) : force(force) {}

    bool areParticlesIdentical(int particle1, int particle2) {
        return force.getParticleType(particle1) ==
               force.getParticleType(particle2);
    }

    int getNumParticleGroups() {
        return force.getNumExceptions() + force.getNumParticles();
    }

    void getParticlesInGroup(int index, std::vector<int> &particles) {
        if (index < force.getNumExceptions()) {
            int particle1, particle2;
            double A, gamma, rCut;
            force.getExceptionParameters(index, particle1, particle2, A, gamma,
                                         rCut);
            particles.resize(2);
            particles[0] = particle1;
            particles[1] = particle2;
        } else {
            particles.clear();
            particles.push_back(index - force.getNumExceptions());
        }
    }

    bool areGroupsIdentical(int group1, int group2) {
        if (group1 < force.getNumExceptions() &&
            group2 < force.getNumExceptions()) {
            int particle1, particle2;
            double A1, gamma1, rCut1, A2, gamma2, rCut2;
            force.getExceptionParameters(group1, particle1, particle2, A1,
                                         gamma1, rCut1);
            force.getExceptionParameters(group2, particle1, particle2, A2,
                                         gamma2, rCut2);
            return A1 == A2 && gamma1 == gamma2 && rCut1 == rCut2;
        }
        return true;
    }

private:
    const OpenMM::DPDForce &force;
};

void OpenMM::CommonCalcDPDForceKernel::initialize(
    const OpenMM::System &system, const OpenMM::DPDForce &force) {
    /* Initialize interactions. */

    OpenMM::ContextSelector selector(cc);
    numParticles = force.getNumParticles();
    int paddedNumParticles{cc.getPaddedNumAtoms()};

    // Create a vector to store vectors of indices of particles that are
    // excluded from interactions with each particle.
    std::vector<std::vector<int>> exclusionList(numParticles);

    // Create a map to associate each particle type to a unique index.
    std::map<int, int> typeIndexMap{{0, 0}};

    // Get and store the type indices corresponding to the particles' types.
    particleTypeIndices.initialize<int>(cc, numParticles,
                                        "dpdParticleTypeIndices");
    std::vector<int> particleTypeIndicesVec(numParticles);
    for (int i{0}; i < numParticles; ++i) {
        int particleType{force.getParticleType(i)};
        auto typeMapping{typeIndexMap.find(particleType)};
        if (typeMapping == typeIndexMap.end()) {
            particleTypeIndicesVec[i] = typeIndexMap.size();
            typeIndexMap[particleType] = typeIndexMap.size();
        } else
            particleTypeIndicesVec[i] = typeMapping.second;
        // Exclude self-interactions.
        exclusionList[i].push_back(i);
    }
    particleTypeIndices.upload(particleTypeIndicesVec);

    // Get and store DPD parameters for each pair of particle types in a
    // row-major 2D array.
    int numTypes{typeIndexMap.size()};
    int numTypePairs{numTypes * (numTypes + 1) / 2};
    pairParams.initialize<OpenMM::mm_float4>(cc, numTypePairs, "dpdPairParams");
    std::vector<OpenMM::mm_float4> pairParamsVec(numTypePairs);
    OpenMM::mm_float4 defaultParams{(float)force.getA(),
                                    (float)force.getGamma(),
                                    (float)force.getRCut(), 0.0f};
    // Give pairs with type 0 the default parameters.
    for (int i{0}; i < numTypes; ++i) pairParamsVec[i] = defaultParams;
    for (int i{0}; i < force.getNumTypePairs(); ++i) {
        int type1, type2;
        double A, gamma, rCut;
        force.getTypePairParameters(i, type1, type2, A, gamma, rCut);
        type1 = typeIndexMap[type1];
        type2 = typeIndexMap[type2];
        pairParamsVec[type1 * numTypes - (type1 * (type1 - 1)) / 2 + type2 -
                      type1] =
            OpenMM::mm_float4((float)A, (float)gamma, (float)rCut, 0.0f);
    }
    pairParams.upload(pairParamsVec);

    /* Record exceptions and exclusions. */

    // Store the indices and pair parameters of particles that interact
    // through exceptions.
    std::vector<std::pair<int, int>> exceptionPairsVec;
    std::vector<OpenMM::mm_float4> exceptionParamsVec;
    for (int i{0}; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double A, gamma, rCut;
        force.getExceptionParameters(i, particle1, particle2, A, gamma, rCut);
        exclusionList[particle1].push_back(particle2);
        exclusionList[particle2].push_back(particle1);
        if (A != 0.0) {
            exceptionParamsVec.push_back(
                mm_float4((float)A, (float)gamma, (float)rCut, 0.0f));
            exceptionPairsVec.push_back(
                std::pair<int, int>(particle1, particle2));
        }
    }
    int numExceptions = exceptionParamsVec.size();
    exceptionPairs.initialize<OpenMM::mm_int4>(cc, std::max(1, numExceptions),
                                               "dpdExceptionPairs");
    exceptionParams.initialize<OpenMM::mm_float4>(
        cc, std::max(1, numExceptions), "dpdExceptionParams");
    if (numExceptions > 0) {
        exceptionPairs.upload(exceptionPairsVec);
        exceptionParams.upload(exceptionParamsVec);
    }

    /* Create the kernels. */

    OpenMM::DPDForce::NonbondedMethod nonbondedMethod =
        force.getNonbondedMethod();
    bool useCutoff =
        (nonbondedMethod != OpenMM::DPDForce::NonbondedMethod::NoCutoff);
    bool usePeriodic =
        (nonbondedMethod == OpenMM::DPDForce::NonbondedMethod::CutoffPeriodic);
    double nonbondedCutoff = force.getCutoffDistance();
    std::map<std::string, std::string> replacements;
    if (useCutoff) {
        replacements["USE_CUTOFF"] = "1";
        if (usePeriodic)
            replacements["USE_PERIODIC"] = "1";
    }

    // TODO: Figure this out. Not implemented yet!
    std::string source =
        cc.replaceStrings(OpenMM::CommonKernelSources::DPDForce, replacements);
    cc.getNonbondedUtilities().addInteraction(
        useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList,
        source, force.getForceGroup(), numParticles > 2000);

    cc.addForce(new OpenMM::CommonCalcDPDForceKernel::ForceInfo(force));
}

void OpenMM::CommonCalcDPDForceKernel::initialize(
    const OpenMM::System &system, const OpenMM::DPDForce &force) {}

void OpenMM::CommonCalcDPDForceKernel::copyParametersToContext(
    OpenMM::ContextImpl &context, const OpenMM::DPDForce &force) {}

void OpenMM::CommonCalcDPDForceKernel::execute(OpenMM::ContextImpl &context,
                                               bool includeForces,
                                               bool includeEnergy,
                                               bool includeConservative) {}