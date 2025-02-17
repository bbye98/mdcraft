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

class OpenMM::CommonCalcDPDForceKernel::ReorderListener
    : public OpenMM::ComputeContext::ReorderListener {
public:
    ReorderListener(OpenMM::CommonCalcDPDForceKernel &owner) : owner(owner) {}
    void execute() { owner.sortAtoms(); }

private:
    OpenMM::CommonCalcDPDForceKernel &owner;
};

void OpenMM::CommonCalcDPDForceKernel::initialize(
    const OpenMM::System &system, const OpenMM::DPDForce &force) {
    // Intialize interactions.

    OpenMM::ContextSelector selector(cc);
    numParticles = force.getNumParticles();
    int paddedNumParticles{cc.getPaddedNumAtoms()};
    std::map<int, int> typeIndexMap{{0, 0}};
    for (const auto &typeNumber : force.getParticleTypes())
        typeIndexMap[typeNumber] = typeIndexMap.size();
    particleTypeIndices.initialize<int>(cc, numParticles,
                                        "dpdParticleTypeIndices");
    std::vector<int> particleTypeIndicesVec(numParticles);
    for (int i = 0; i < numParticles; i++)
        particleTypeIndicesVec[i] = typeIndexMap[force.getParticleType(i)];
    particleTypeIndices.upload(particleTypeIndicesVec);

    int numTypes{typeIndexMap.size()};
    int numTypePairs{numTypes * (numTypes + 1) / 2};
    pairParams.initialize<OpenMM::mm_float4>(cc, numTypePairs, "dpdPairParams");
    std::vector<OpenMM::mm_float4> pairParamsVec(numTypePairs);
    OpenMM::mm_float4 defaultParams{(float)force.getA(),
                                    (float)force.getGamma(),
                                    (float)force.getRCut(), 0.0f};
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

    // Record exceptions and exclusions.

    std::vector<OpenMM::mm_float4> exceptionParamsVec;
    for (int i{0}; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double A, gamma, rCut;
        force.getExceptionParameters(i, particle1, particle2, A, gamma, rCut);
        excludedPairs.push_back(std::pair<int, int>(particle1, particle2));
        if (A != 0.0) {
            exceptionParamsVec.push_back(
                mm_float4((float)A, (float)gamma, (float)rCut, 0.0f));
            exceptionPairs.push_back(std::pair<int, int>(particle1, particle2));
        }
    }
    int numExceptions = exceptionParamsVec.size();
    exclusions.initialize<int>(cc, std::max(1, (int)excludedPairs.size()),
                               "dpdExclusions");
    exclusionStartIndex.initialize<int>(cc, numParticles + 1,
                                        "dpdExclusionStartIndex");
    exceptionParticles.initialize<OpenMM::mm_int4>(
        cc, std::max(1, numExceptions), "dpdExceptionParticles");
    exceptionParams.initialize<OpenMM::mm_float2>(
        cc, std::max(1, numExceptions), "dpdExceptionParams");
    if (numExceptions > 0)
        exceptionParams.upload(exceptionParamsVec);

    // Create data structures used for the neighbor list.

    int numAtomBlocks = (numParticles + 31) / 32;
    int elementSize =
        (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    blockCenter.initialize(cc, numAtomBlocks, 4 * elementSize,
                           "dpdBlockCenter");
    blockBoundingBox.initialize(cc, numAtomBlocks, 4 * elementSize,
                                "dpdBlockBoundingBox");
    sortedPositions.initialize(cc, numParticles, 4 * elementSize,
                               "dpdSortedPositions");
    maxNeighborBlocks = numParticles * 2;
    neighbors.initialize<int>(cc, maxNeighborBlocks * 32, "dpdNeighbors");
    neighborIndex.initialize<int>(cc, maxNeighborBlocks, "dpdNeighborIndex");
    neighborBlockCount.initialize<int>(cc, 1, "dpdNeighborBlockCount");
    event = cc.createEvent();

    // Create the kernels.

    OpenMM::DPDForce::NonbondedMethod nonbondedMethod =
        force.getNonbondedMethod();
    bool useCutoff =
        (nonbondedMethod != OpenMM::DPDForce::NonbondedMethod::NoCutoff);
    bool usePeriodic =
        (nonbondedMethod == OpenMM::DPDForce::NonbondedMethod::CutoffPeriodic);
    double nonbondedCutoff = force.getCutoffDistance();
    std::map<std::string, std::string> defines;
    if (useCutoff) {
        defines["USE_CUTOFF"] = "1";
        if (usePeriodic)
            defines["USE_PERIODIC"] = "1";
    }
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    OpenMM::ComputeProgram program =
        cc.compileProgram(OpenMM::CommonKernelSources::DPDForce, defines);
    // TODO: Register kernels here.
    cc.addForce(new OpenMM::CommonCalcDPDForceKernel::ForceInfo(force));
    cc.addReorderListener(new ReorderListener(*this));
}

void OpenMM::CommonCalcDPDForceKernel::initialize(
    const OpenMM::System &system, const OpenMM::DPDForce &force) {}

void OpenMM::CommonCalcDPDForceKernel::copyParametersToContext(
    OpenMM::ContextImpl &context, const OpenMM::DPDForce &force) {}

void OpenMM::CommonCalcDPDForceKernel::execute(OpenMM::ContextImpl &context,
                                               bool includeForces,
                                               bool includeEnergy,
                                               bool includeConservative) {}

void OpenMM::CommonCalcDPDForceKernel::sortAtoms() {
    // Sort the list of atoms by type to avoid thread divergence. This is
    // executed every time the atoms are reordered.

    int nextIndex{0};
    std::vector<int> particles(cc.getPaddedNumAtoms(), 0);
    const std::vector<int> &order{cc.getAtomIndex()};
    std::vector<int> inverseOrder(order.size(), -1);
    for (int i = 0; i < cc.getNumAtoms(); i++) {
        int atom{order[i]};
        inverseOrder[atom] = nextIndex;
        particles[nextIndex++] = atom;
    }
    sortedParticles.upload(particles);

    // Update the list of exception particles.

    int numExceptions { exceptionPairs.size(); }
    if (numExceptions > 0) {
        std::vector<OpenMM::mm_int4> exceptionParticlesVec(numExceptions);
        for (int i{0}; i < numExceptions; ++i)
            exceptionParticlesVec[i] = OpenMM::mm_int4(
                exceptionAtoms[i].first, exceptionAtoms[i].second,
                inverseOrder[exceptionAtoms[i].first],
                inverseOrder[exceptionAtoms[i].second]);
        exceptionParticles.upload(exceptionParticlesVec);
    }

    // Rebuild the list of exclusions.

    std::vector<std::vector<int>> excludedAtoms(numParticles);
    for (int i{0}; i < excludedPairs.size(); ++i) {
        int first{inverseOrder[std::min(excludedPairs[i].first,
                                        excludedPairs[i].second)]};
        int second{inverseOrder[std::max(excludedPairs[i].first,
                                         excludedPairs[i].second)]};
        excludedAtoms[first].push_back(second);
    }
    int index{0};
    std::vector<int> exclusionVec(exclusions.getSize());
    std::vector<int> startIndexVec(exclusionStartIndex.getSize());
    for (int i{0}; i < numRealParticles; ++i) {
        startIndexVec[i] = index;
        for (int j{0}; j < excludedAtoms[i].size(); ++j)
            exclusionVec[index++] = excludedAtoms[i][j];
    }
    startIndexVec[numRealParticles] = index;
    exclusions.upload(exclusionVec);
    exclusionStartIndex.upload(startIndexVec);
}