#include "openmm/DPDForce.h"

#include <algorithm>
#include <map>
#include <utility>

#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/DPDForceImpl.h"

OpenMM::DPDForce::DPDForce(double A, double gamma, double rCut, double cutoff,
                           bool conservative) {
    defaultA = A;
    setGamma(gamma);
    setRCut(rCut);
    setCutoffDistance(cutoff);
    includeConservative = conservative;
}

void OpenMM::DPDForce::setGamma(double gamma) {
    if (gamma < 0.0)
        throw OpenMM::OpenMMException(
            "DPDForce: gamma must be greater than or equal to 0");
    defaultGamma = gamma;
}

void OpenMM::DPDForce::setRCut(double rCut) {
    if (rCut <= 0.0)
        throw OpenMM::OpenMMException("DPDForce: rCut must be greater than 0");
    defaultRCut = rCut;
}

void OpenMM::DPDForce::setCutoffDistance(double cutoff) {
    if (cutoff <= 0.0)
        throw OpenMM::OpenMMException(
            "DPDForce: cutoff must be greater than 0");
    cutoffDistance = cutoff;
}

int OpenMM::DPDForce::addParticle(int typeIndex) {
    particleTypes.push_back(typeIndex);
    return particleTypes.size() - 1;
}

int OpenMM::DPDForce::getParticleType(int particleIndex) const {
    ASSERT_VALID_INDEX(particleIndex, particleTypes);
    return particleTypes[particleIndex];
}

void OpenMM::DPDForce::setParticleType(int particleIndex, int typeIndex) {
    ASSERT_VALID_INDEX(particleIndex, particleTypes);

    if (particleTypes[particleIndex] == typeIndex)
        return;

    particleTypes[particleIndex] = typeIndex;
    if (numContexts > 0) {
        firstChangedParticle = std::min(firstChangedParticle, particleIndex);
        lastChangedParticle = std::max(lastChangedParticle, particleIndex);
    }
}

int OpenMM::DPDForce::addTypePair(int type1, int type2, double A, double gamma,
                                  double rCut, bool replace = false) {
    if (type1 == 0 || type2 == 0)
        throw OpenMM::OpenMMException(
            "DPDForce.addTypePair: Particle type cannot be 0");
    if (gamma < 0.0)
        throw OpenMM::OpenMMException(
            "DPDForce.addTypePair: gamma must be greater than or equal to 0");
    if (rCut <= 0.0)
        throw OpenMM::OpenMMException(
            "DPDForce.addTypePair: rCut must be greater than 0");

    auto types = std::minmax(type1, type2);
    auto [iter, inserted] = typePairMap.emplace(types, typePairs.size());
    if (inserted)
        typePairs.emplace_back(
            TypePairInfo(types.first, types.second, A, gamma, rCut));
    else if (replace) {
        auto &typePair = typePairs[iter->second];
        typePair.A = A;
        typePair.gamma = gamma;
        typePair.rCut = rCut;
    } else
        throw OpenMM::OpenMMException(
            "DPDForce.addTypePair: There are already parameters defined for "
            "particles of types " +
            std::to_string(types.first) + " and " +
            std::to_string(types.second));
    return iter->second;
}

int OpenMM::DPDForce::getTypePairIndex(int type1, int type2) const {
    if (type1 == 0 || type2 == 0)
        return -1;
    auto iter = typePairMap.find(std::minmax(type1, type2));
    return (iter == typePairMap.end()) ? -1 : iter->second;
}

void OpenMM::DPDForce::getTypePairParameters(int index, int &type1, int &type2,
                                             double &A, double &gamma,
                                             double &rCut) const {
    ASSERT_VALID_INDEX(index, typePairs);
    type1 = typePairs[index].type1;
    type2 = typePairs[index].type2;
    A = typePairs[index].A;
    gamma = typePairs[index].gamma;
    rCut = typePairs[index].rCut;
}

void OpenMM::DPDForce::setTypePairParameters(int index, int type1, int type2,
                                             double A, double gamma,
                                             double rCut) {
    ASSERT_VALID_INDEX(index, typePairs);
    if (type1 == 0 || type2 == 0)
        throw OpenMM::OpenMMException(
            "DPDForce.addTypePair: Particle type cannot be 0");
    if (gamma < 0.0)
        throw OpenMM::OpenMMException(
            "DPDForce.setTypePairParameters: gamma must be greater than or "
            "equal to 0");
    if (rCut <= 0.0)
        throw OpenMM::OpenMMException(
            "DPDForce.setTypePairParameters: rCut must be greater than 0");

    TypePairInfo &typePair = typePairs[index];
    std::tie(type1, type2) = std::minmax(type1, type2);
    if (typePair.type1 == type1 && typePair.type2 == type2 && typePair.A == A &&
        typePair.gamma == gamma && typePair.rCut == rCut)
        return;

    typePair.type1 = type1;
    typePair.type2 = type2;
    typePair.A = A;
    typePair.gamma = gamma;
    typePair.rCut = rCut;

    if (numContexts > 0) {
        firstChangedParticle = std::min(firstChangedParticle, index);
        lastChangedParticle = std::max(lastChangedParticle, index);
    }
}

int OpenMM::DPDForce::addException(int particle1, int particle2, double A,
                                   double gamma, double rCut,
                                   bool replace = false) {
    if (gamma < 0.0)
        throw OpenMM::OpenMMException(
            "DPDForce.addException: gamma must be greater than or equal to 0");
    if (rCut <= 0.0)
        throw OpenMM::OpenMMException(
            "DPDForce.addException: rCut must be greater than 0");

    auto particles = std::minmax(particle1, particle2);
    auto [iter, inserted] = exceptionMap.emplace(particles, exceptions.size());
    if (inserted)
        exceptions.emplace_back(
            ExceptionInfo(particle1, particle2, A, gamma, rCut));
    else if (replace) {
        auto &exception = exceptions[iter->second];
        exception.A = A;
        exception.gamma = gamma;
        exception.rCut = rCut;
    } else
        throw OpenMM::OpenMMException(
            "DPDForce.addException: There is already an exception for "
            "particles " +
            std::to_string(particle1) + " and " + std::to_string(particle2));
    return iter->second;
}

void OpenMM::DPDForce::getExceptionParameters(int exceptionIndex,
                                              int &particle1, int &particle2,
                                              double &A, double &gamma,
                                              double &rCut) const {
    ASSERT_VALID_INDEX(exceptionIndex, exceptions);
    particle1 = exceptions[exceptionIndex].particle1;
    particle2 = exceptions[exceptionIndex].particle2;
    A = exceptions[exceptionIndex].A;
    gamma = exceptions[exceptionIndex].gamma;
    rCut = exceptions[exceptionIndex].rCut;
}

void OpenMM::DPDForce::setExceptionParameters(int exceptionIndex, int particle1,
                                              int particle2, double A,
                                              double gamma, double rCut) {
    ASSERT_VALID_INDEX(exceptionIndex, exceptions);
    if (gamma < 0.0)
        throw OpenMM::OpenMMException(
            "DPDForce.setExceptionParameters: gamma must be greater than "
            "or equal to 0");
    if (rCut <= 0.0)
        throw OpenMM::OpenMMException(
            "DPDForce.setExceptionParameters: rCut must be greater than 0");

    ExceptionInfo &exception = exceptions[exceptionIndex];

    if (exception.particle1 == particle1 && exception.particle2 == particle2 &&
        exception.A == A && exception.gamma == gamma && exception.rCut == rCut)
        return;

    exception.particle1 = particle1;
    exception.particle2 = particle2;
    exception.A = A;
    exception.gamma = gamma;
    exception.rCut = rCut;

    if (numContexts > 0) {
        firstChangedException = std::min(firstChangedException, exceptionIndex);
        lastChangedException = std::max(lastChangedException, exceptionIndex);
    }
}

void OpenMM::DPDForce::createExceptionsFromBonds(
    const std::vector<std::pair<int, int>> &bonds, double A14Scale) {
    for (const std::pair<int, int> &bond : bonds)
        if (bond.first < 0 || bond.second < 0 ||
            bond.first >= particleTypes.size() ||
            bond.second >= particleTypes.size())
            throw OpenMM::OpenMMException(
                "DPDForce.createExceptionsFromBonds: Illegal particle "
                "index in list of bonds");

    std::vector<std::set<int>> exclusions(particleTypes.size());
    std::vector<std::set<int>> bonded12(exclusions.size());
    for (const std::pair<int, int> &bond : bonds) {
        bonded12[bond.first].insert(bond.second);
        bonded12[bond.second].insert(bond.first);
    }
    for (int i = 0; i < (int)exclusions.size(); ++i)
        addExclusionsToSet(bonded12, exclusions[i], i, i, 2);

    for (int i = 0; i < (int)exclusions.size(); ++i) {
        std::set<int> bonded13;
        addExclusionsToSet(bonded12, bonded13, i, i, 1);
        for (int j : exclusions[i]) {
            if (j < i) {
                if (bonded13.find(j) == bonded13.end()) {
                    const int type1 = particleTypes[i];
                    const int type2 = particleTypes[j];
                    if (type1 != 0 && type2 != 0) {
                        auto iter = typePairMap.find(std::minmax(type1, type2));
                        if (iter == typePairMap.end()) {
                            throw OpenMM::OpenMMException(
                                "DPDForce.createExceptionsFromBonds: No DPD "
                                "parameters defined for particles of types " +
                                std::to_string(type1) + " and " +
                                std::to_string(type2));
                        }
                        int typePairIndex = iter->second;
                        addException(j, i,
                                     A14Scale * typePairs[typePairIndex].A,
                                     typePairs[typePairIndex].gamma,
                                     typePairs[typePairIndex].rCut);
                    } else
                        addException(j, i, A14Scale * defaultA, defaultGamma,
                                     defaultRCut);
                } else
                    addException(j, i, 0.0, 0.0, 0.0);
            }
        }
    }
}

void OpenMM::DPDForce::addExclusionsToSet(
    const std::vector<std::set<int>> &bonded12, std::set<int> &exclusions,
    int baseParticle, int fromParticle, int currentLevel) const {
    for (int i : bonded12[fromParticle]) {
        if (i != baseParticle)
            exclusions.insert(i);
        if (currentLevel > 0)
            addExclusionsToSet(bonded12, exclusions, baseParticle, i,
                               currentLevel - 1);
    }
}

void OpenMM::DPDForce::updateParametersInContext(Context &context) {
    dynamic_cast<DPDForceImpl &>(getImplInContext(context))
        .updateParametersInContext(getContextImpl(context),
                                   firstChangedParticle, lastChangedParticle,
                                   firstChangedException, lastChangedException);
    if (numContexts == 1) {
        firstChangedParticle = particleTypes.size();
        lastChangedParticle = -1;
        firstChangedException = exceptions.size();
        lastChangedException = -1;
    }
}

OpenMM::ForceImpl *OpenMM::DPDForce::createImpl() const {
    if (numContexts == 0) {
        firstChangedParticle = particleTypes.size();
        lastChangedParticle = -1;
        firstChangedException = exceptions.size();
        lastChangedException = -1;
    }
    numContexts++;
    return new OpenMM::DPDForceImpl(*this);
}