#include "openmm/internal/DPDForceImpl.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <unordered_set>

#include "openmm/CalcDPDForceKernel.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/Messages.h"

OpenMM::DPDForceImpl::DPDForceImpl(const OpenMM::DPDForce &owner)
    : owner(owner) {
    forceGroup = owner.getForceGroup();
    includeConservative = owner.getIncludeConservative();
}

OpenMM::DPDForceImpl::~DPDForceImpl() {}

void OpenMM::DPDForceImpl::initialize(ContextImpl &context) {
    kernel = context.getPlatform().createKernel(
        OpenMM::CalcDPDForceKernel::Name(), context);

    const System &system = context.getSystem();
    if (owner.getNumParticles() != system.getNumParticles())
        throw OpenMM::OpenMMException(
            "DPDForce must have exactly as many particles as the System it "
            "belongs to.");

    std::unordered_set<int> uniqueTypes;
    for (int i = 0; i < owner.getNumParticles(); i++) {
        int typeIndex = owner.getParticleType(i);
        if (typeIndex != 0)
            uniqueTypes.insert(typeIndex);
    }
    std::vector<int> uniqueTypeVector(uniqueTypes.begin(), uniqueTypes.end());
    for (int i = 0; i < uniqueTypeVector.size(); i++) {
        int type1 = uniqueTypeVector[i];
        if (type1 == 0)
            continue;
        for (int j = i; j < uniqueTypeVector.size(); j++) {
            int type2 = uniqueTypeVector[j];
            if (type2 == 0)
                continue;
            std::tie(type1, type2) = std::minmax(type1, type2);
            if (owner.getTypePairIndex(type1, type2) == -1) {
                throw OpenMM::OpenMMException(
                    "DPDForce: No DPD parameters defined for particles of "
                    "types " +
                    std::to_string(type1) + " and " + std::to_string(type2));
            }
        }
    }

    std::vector<std::set<int>> exceptions(owner.getNumParticles());
    for (int i = 0; i < owner.getNumExceptions(); i++) {
        int particle[2];
        double A, gamma, rCut;
        owner.getExceptionParameters(i, particle[0], particle[1], A, gamma,
                                     rCut);
        int minp = std::min(particle[0], particle[1]);
        int maxp = std::max(particle[0], particle[1]);
        for (int j = 0; j < 2; j++) {
            if (particle[j] < 0 || particle[j] >= owner.getNumParticles()) {
                throw OpenMM::OpenMMException(
                    "DPDForce: Illegal particle index for an exception: " +
                    std::to_string(particle[j]));
            }
        }
        if (exceptions[minp].count(maxp) > 0) {
            throw OpenMM::OpenMMException(
                "DPDForce: Multiple exceptions are specified for particles " +
                std::to_string(particle[0]) + " and " +
                std::to_string(particle[1]));
        }
        exceptions[minp].insert(maxp);
        if (gamma < 0)
            throw OpenMM::OpenMMException(
                "DPDForce: gamma for an exception cannot be negative");
        if (rCut < 0)
            throw OpenMM::OpenMMException(
                "DPDForce: rCut for an exception cannot be negative");
    }
}

std::vector<std::string> OpenMM::DPDForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(OpenMM::CalcDPDForceKernel::Name());
    return names;
}

std::map<std::string, double> OpenMM::DPDForceImpl::getDefaultParameters() {
    std::map<std::string, double> parameters;
    parameters["A"] = owner.getA();
    parameters["gamma"] = owner.getGamma();
    parameters["rCut"] = owner.getRCut();
    return parameters;
}

void OpenMM::DPDForceImpl::updateParametersInContext(ContextImpl &context,
                                                     int firstParticle,
                                                     int lastParticle,
                                                     int firstException,
                                                     int lastException) {
    kernel.getAs<OpenMM::CalcDPDForceKernel>().copyParametersToContext(
        context, owner, firstParticle, lastParticle, firstException,
        lastException);
    context.systemChanged();
}

double OpenMM::DPDForceImpl::calcForcesAndEnergy(ContextImpl &context,
                                                 bool includeForces,
                                                 bool includeEnergy,
                                                 int groups) {
    if ((groups & (1 << forceGroup)) != 0)
        return kernel.getAs<OpenMM::CalcDPDForceKernel>().execute(
            context, includeForces, includeEnergy, includeConservative);
    return 0.0;
}