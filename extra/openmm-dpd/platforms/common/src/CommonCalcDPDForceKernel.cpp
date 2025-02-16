#include "openmm/common/CommonCalcDPDForceKernel.h"

#include <map>

#include "openmm/common/ComputeParameterSet.h"
#include "openmm/common/ContextSelector.h"

void OpenMM::CommonCalcDPDForceKernel::initialize(const System &system,
                                                  const DPDForce &force) {
    OpenMM::ContextSelector selector(cc);
    int numContexts = cc.getNumContexts();
    int startIndex =
        cc.getContextIndex() * force.getNumParticles() / numContexts;
    int endIndex =
        (cc.getContextIndex() + 1) * force.getNumParticles() / numContexts;
    numParticles = endIndex - startIndex;
    if (numParticles == 0)
        return;

    std::map<int, int> typeIndexMap{{0, 0}};
    for (const auto &typeNumber : force.getUniqueParticleTypes())
        typeIndexMap[typeNumber] = typeIndexMap.size();

    particleTypeIndices.initialize<int>(cc, numParticles,
                                        "dpdParticleTypeIndices");
    std::vector<int> particleTypeIndicesVec(numParticles);
    for (int i = 0; i < numParticles; i++)
        particleTypeIndicesVec[i] =
            typeIndexMap[force.getParticleType(i + startIndex)];
    particleTypeIndices.upload(particleTypeIndicesVec);

    numTypes = typeIndexMap.size();
    int numTypePairs = numTypes * (numTypes + 1) / 2;
    pairParams.initialize<OpenMM::mm_float4>(cc, numTypePairs, "dpdPairParams");
    std::vector<OpenMM::mm_float4> pairParamsVec(numTypePairs);
    OpenMM::mm_float4 defaultParams{(float)force.getA(),
                                    (float)force.getGamma(),
                                    (float)force.getRCut(), 0.0f};
    for (int i = 0; i < numTypes; ++i) pairParamsVec[i] = defaultParams;
    for (int i = 0; i < force.getNumTypePairs(); ++i) {
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

    // TODO: Handle exceptions.
    // TODO: Set up replacement dictionary.
}