#include "ReferenceCalcDPDForceKernel.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "SimTKOpenMMUtilities.h"
#include "openmm/reference/ReferenceForce.h"

static constexpr double EPSILON = 1.0e-12;

OpenMM::ReferenceCalcDPDForceKernel::~ReferenceCalcDPDForceKernel() {
    if (neighborList != nullptr) {
        delete neighborList;
        neighborList = nullptr;
    }
}

void OpenMM::ReferenceCalcDPDForceKernel::initialize(const System& system,
                                                     const DPDForce& force) {
    defaultA = force.getA();
    defaultGamma = force.getGamma();
    defaultRCut = force.getRCut();
    temperature = force.getTemperature();

    numParticles = force.getNumParticles();
    particleTypes.resize(numParticles);
    for (int i = 0; i < numParticles; ++i)
        particleTypes[i] = force.getParticleType(i);

    for (const auto& typeNumber : force.getUniqueParticleTypes())
        typeIndexMap[typeNumber] = typeIndexMap.size();
    numTypes = typeIndexMap.size();
    pairParams.resize(numTypes * (numTypes + 1) / 2);
    for (int i = 0; i < force.getNumTypePairs(); ++i) {
        int type1, type2, pairIndex;
        double A, gamma, rCut;
        force.getTypePairParameters(i, type1, type2, A, gamma, rCut);
        type1 = typeIndexMap[type1];
        type2 = typeIndexMap[type2];
        pairParams[type1 * numTypes - (type1 * (type1 - 1)) / 2 + type2 -
                   type1] = {A, gamma, rCut};
    }

    std::vector<int> exceptionIndices;
    perParticleExclusions.resize(numParticles);
    for (int i = 0; i < force.getNumExceptions(); ++i) {
        int particle1, particle2;
        double A, gamma, rCut;
        force.getExceptionParameters(i, particle1, particle2, A, gamma, rCut);
        perParticleExclusions[particle1].insert(particle2);
        perParticleExclusions[particle2].insert(particle1);
        if (A != 0)
            exceptionIndices.push_back(i);
    }

    numExceptions = exceptionIndices.size();
    numTotalExceptions = force.getNumExceptions();
    exceptionParticlePairs.resize(numExceptions);
    exceptionParams.resize(numExceptions);
    for (int i = 0; i < numExceptions; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(
            exceptionIndices[i], exceptionParticlePairs[i][0],
            exceptionParticlePairs[i][1], exceptionParams[i][0],
            exceptionParams[i][1], exceptionParams[i][2]);
    }

    dpdMethod = CalcDPDForceKernel::DPDMethod(force.getDPDMethod());
    nonbondedCutoff = force.getCutoffDistance();
    if (dpdMethod == NoCutoff)
        neighborList = nullptr;
    else
        neighborList = new NeighborList();
    if (dpdMethod == CutoffPeriodic)
        exceptionsArePeriodic =
            force.getExceptionsUsePeriodicBoundaryConditions();
    else
        exceptionsArePeriodic = false;
}

void OpenMM::ReferenceCalcDPDForceKernel::copyParametersToContext(
    ContextImpl& context, const DPDForce& force) {
    if (force.getNumParticles() != numParticles)
        throw OpenMMException(
            "DPDForce.updateParametersInContext: The number of particles has "
            "changed");

    defaultA = force.getA();
    defaultGamma = force.getGamma();
    defaultRCut = force.getRCut();
    temperature = force.getTemperature();

    std::set<int> uniqueTypesSet;
    for (int i = 0; i < numParticles; i++) {
        particleTypes[i] = force.getParticleType(i);
        if (particleTypes[i] != 0)
            uniqueTypesSet.insert(particleTypes[i]);
    }
    std::vector<int> uniqueTypesVector(uniqueTypesSet.begin(),
                                       uniqueTypesSet.end());
    for (int i = 0; i < uniqueTypesVector.size(); ++i) {
        int type1 = uniqueTypesVector[i];
        if (type1 == 0)
            continue;
        for (int j = i; j < uniqueTypesVector.size(); ++j) {
            int type2 = uniqueTypesVector[j];
            if (type2 == 0)
                continue;
            if (force.getTypePairIndex(type1, type2) == -1) {
                throw OpenMM::OpenMMException(
                    "DPDForce: No DPD parameters defined for particles of "
                    "types " +
                    std::to_string(type1) + " and " + std::to_string(type2));
            }
        }
    }

    for (const auto& typeNumber : uniqueTypesSet)
        typeIndexMap[typeNumber] = typeIndexMap.size();
    numTypes = typeIndexMap.size();
    pairParams.resize(numTypes * (numTypes + 1) / 2);
    for (int i = 0; i < force.getNumTypePairs(); ++i) {
        int type1, type2;
        double A, gamma, rCut;
        force.getTypePairParameters(i, type1, type2, A, gamma, rCut);
        type1 = typeIndexMap[type1];
        type2 = typeIndexMap[type2];
        pairParams[type1 * numTypes - (type1 * (type1 - 1)) / 2 + type2 -
                   type1] = {A, gamma, rCut};
    }

    std::vector<int> exceptionIndices;
    perParticleExclusions.clear();
    perParticleExclusions.resize(numParticles);
    for (int i = 0; i < force.getNumExceptions(); ++i) {
        int particle1, particle2;
        double A, gamma, rCut;
        force.getExceptionParameters(i, particle1, particle2, A, gamma, rCut);
        perParticleExclusions[particle1].insert(particle2);
        perParticleExclusions[particle2].insert(particle1);
        if (A != 0)
            exceptionIndices.push_back(i);
    }

    numExceptions = exceptionIndices.size();
    numTotalExceptions = force.getNumExceptions();
    exceptionParticlePairs.clear();
    exceptionParticlePairs.resize(numExceptions);
    exceptionParams.clear();
    exceptionParams.resize(numExceptions);
    for (int i = 0; i < numExceptions; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(
            exceptionIndices[i], exceptionParticlePairs[i][0],
            exceptionParticlePairs[i][1], exceptionParams[i][0],
            exceptionParams[i][1], exceptionParams[i][2]);
    }
}

double OpenMM::ReferenceCalcDPDForceKernel::execute(ContextImpl& context,
                                                    bool includeForces,
                                                    bool includeEnergy,
                                                    bool includeConservative) {
    ReferencePlatform::PlatformData* data =
        reinterpret_cast<ReferencePlatform::PlatformData*>(
            context.getPlatformData());
    std::vector<Vec3>& positions = *data->positions;
    std::vector<Vec3>& velocities = *data->velocities;
    std::vector<Vec3>& forces = *data->forces;

    bool cutoff{dpdMethod != NoCutoff};
    bool periodic{dpdMethod == CutoffPeriodic};
    OpenMM::Vec3* boxVectors{data->periodicBoxVectors};
    if (dpdMethod != NoCutoff)
        computeNeighborListVoxelHash(*neighborList, numParticles, positions,
                                     perParticleExclusions, boxVectors,
                                     periodic, nonbondedCutoff);
    if (periodic) {
        double minAllowedSize = 1.999999 * nonbondedCutoff;
        if (boxVectors[0][0] < minAllowedSize ||
            boxVectors[1][1] < minAllowedSize ||
            boxVectors[2][2] < minAllowedSize)
            throw OpenMM::OpenMMException(
                "The periodic box size has decreased to less than twice the "
                "nonbonded cutoff.");
    }

    double dt{context.getIntegrator().getStepSize()};
    double totalEnergy{0.0};
    if (cutoff)
        for (auto& pair : *neighborList) {
            calculateOneIxn(pair.first, pair.second, positions, velocities,
                            forces, totalEnergy, dt, includeConservative,
                            periodic, boxVectors);
        }
    else
        for (int ii = 0; ii < numParticles; ii++) {
            for (int jj = ii + 1; jj < numParticles; jj++)
                if (perParticleExclusions[jj].find(ii) ==
                    perParticleExclusions[jj].end()) {
                    calculateOneIxn(ii, jj, positions, velocities, forces,
                                    totalEnergy, dt, includeConservative,
                                    periodic, boxVectors);
                }
        }

    for (int i = 0; i < numExceptions; ++i) {
        calculateOneIxn(exceptionParticlePairs[i][0],
                        exceptionParticlePairs[i][1], positions, velocities,
                        forces, totalEnergy, dt, includeConservative,
                        exceptionsArePeriodic, boxVectors, &exceptionParams[i]);
    }

    return totalEnergy;
}

void OpenMM::ReferenceCalcDPDForceKernel::calculateOneIxn(
    int ii, int jj, const std::vector<OpenMM::Vec3>& positions,
    const std::vector<OpenMM::Vec3>& velocities,
    std::vector<OpenMM::Vec3>& forces, double& totalEnergy, const double dt,
    bool includeConservative, bool periodic, const OpenMM::Vec3* boxVectors,
    const std::array<double, 3>* params) {
    double A, gamma, rCut;
    if (params != nullptr) {
        A = (*params)[0];
        gamma = (*params)[1];
        rCut = (*params)[2];
    } else {
        int type1 = particleTypes[ii];
        int type2 = particleTypes[jj];
        if (type1 == 0 || type2 == 0) {
            A = defaultA;
            gamma = defaultGamma;
            rCut = defaultRCut;
        } else {
            if (type1 > type2)
                std::swap(type1, type2);
            type1 = typeIndexMap[type1];
            type2 = typeIndexMap[type2];
            int pairParamsIndex =
                type1 * numTypes - (type1 * (type1 - 1)) / 2 + type2 - type1;
            A = pairParams[pairParamsIndex][0];
            gamma = pairParams[pairParamsIndex][1];
            rCut = pairParams[pairParamsIndex][2];
        }
    }

    double dr[ReferenceForce::LastDeltaRIndex];
    if (periodic)
        OpenMM::ReferenceForce::getDeltaRPeriodic(positions[ii], positions[jj],
                                                  boxVectors, dr);
    else
        OpenMM::ReferenceForce::getDeltaR(positions[ii], positions[jj], dr);

    double r = dr[OpenMM::ReferenceForce::RIndex];
    bool overlap = r < EPSILON;
    double weight, weight2;
    OpenMM::Vec3 drUnitVector;
    if (overlap) {
        weight = 1.0;
        weight2 = 1.0;
    } else {
        weight = 1.0 - r / rCut;
        weight2 = weight * weight;
        drUnitVector = {dr[OpenMM::ReferenceForce::XIndex] / r,
                        dr[OpenMM::ReferenceForce::YIndex] / r,
                        dr[OpenMM::ReferenceForce::ZIndex] / r};
    }
    OpenMM::Vec3 dv = velocities[ii] - velocities[jj];
    double sigma = sqrt(2 * gamma * BOLTZ * temperature);

    if (r < rCut) {
        if (!overlap) {
            double forceMag =
                -gamma * weight2 * drUnitVector.dot(dv) +
                sigma * weight *
                    SimTKOpenMMUtilities::getNormallyDistributedRandomNumber() /
                    sqrt(dt);
            if (includeConservative)
                forceMag += A * weight;
            for (int kk = 0; kk < 3; ++kk) {
                double fkk = forceMag * drUnitVector[kk];
                forces[ii][kk] += fkk;
                forces[jj][kk] -= fkk;
            }
        }

        if (includeConservative && totalEnergy)
            totalEnergy += 0.5 * A * rCut * weight2;
    }
}