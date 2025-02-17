#ifndef OPENMM_DPDFORCE_H_
#define OPENMM_DPDFORCE_H_

#include <set>
#include <unordered_map>
#include <vector>

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "openmm/internal/windowsExport.h"

namespace OpenMM {

    class OPENMM_EXPORT DPDForce : public Force {
    public:
        enum NonbondedMethod {
            NoCutoff = 0,
            CutoffNonPeriodic = 1,
            CutoffPeriodic = 2
        };

        DPDForce(double A = 0.0, double gamma = 0.0, double rCut = 0.0,
                 double temperature = 300, double cutoff = 0.0,
                 bool conservative = true);

        NonbondedMethod getNonbondedMethod() const { return nonbondedMethod; }

        void setNonbondedMethod(NonbondedMethod method) {
            nonbondedMethod = method;
        }

        double getA() const { return defaultA; }

        void setA(double A) { defaultA = A; }

        double getGamma() const { return defaultGamma; }

        void setGamma(double gamma);

        double getRCut() const { return defaultRCut; }

        void setRCut(double rCut);

        void setTemperature(double temperature);

        double getTemperature() const { return temperature; }

        double getCutoffDistance() const { return nonbondedCutoff; }

        void setCutoffDistance(double cutoff);

        int getNumParticles() const { return particleTypes.size(); }

        int addParticle(int typeID = 0);

        int getParticleType(int particleIndex) const;

        void setParticleType(int particleIndex, int typeID);

        std::set<int> getParticleTypes() const;

        int getNumTypes() const { return getParticleTypes().size(); }

        int getNumTypePairs() const { return typePairs.size(); }

        int addTypePair(int type1, int type2, double A, double gamma,
                        double rCut, bool replace = false);

        int getTypePairIndex(int type1, int type2) const;

        void getTypePairParameters(int typePairIndex, int &type1, int &type2,
                                   double &A, double &gamma,
                                   double &rCut) const;

        void setTypePairParameters(int typePairIndex, int type1, int type2,
                                   double A, double gamma, double rCut);

        int getNumExceptions() const { return exceptions.size(); }

        int addException(int particle1, int particle2, double A, double gamma,
                         double rCut, bool replace = false);

        void getExceptionParameters(int exceptionIndex, int &particle1,
                                    int &particle2, double &A, double &gamma,
                                    double &rCut) const;

        void setExceptionParameters(int exceptionIndex, int particle1,
                                    int particle2, double A, double gamma,
                                    double rCut);

        void createExceptionsFromBonds(
            const std::vector<std::pair<int, int>> &bonds, double A14Scale);

        bool getIncludeConservative() const { return includeConservative; }

        void setIncludeConservative(bool include) {
            includeConservative = include;
        }

        void updateParametersInContext(Context &context);

        bool usesPeriodicBoundaryConditions() const {
            return nonbondedMethod == NonbondedMethod::CutoffPeriodic;
        }

        bool getExceptionsUsePeriodicBoundaryConditions() const {
            return exceptionsUsePeriodic;
        }

        void setExceptionsUsePeriodicBoundaryConditions(bool periodic) {
            exceptionsUsePeriodic = periodic;
        }

    protected:
        ForceImpl *createImpl() const;

    private:
        struct PairHash {
            std::size_t operator()(const std::pair<int, int> &p) const {
                return std::hash<int>()(p.first) ^ std::hash<int>()(p.second);
            }
        };

        struct TypePairInfo {
            int type1, type2;
            double A, gamma, rCut;

            TypePairInfo(int t1, int t2, double a, double g, double r)
                : type1(t1), type2(t2), A(a), gamma(g), rCut(r) {}
        };

        struct ExceptionInfo {
            int particle1, particle2;
            double A, gamma, rCut;

            ExceptionInfo(int p1, int p2, double a, double g, double r)
                : particle1(p1), particle2(p2), A(a), gamma(g), rCut(r) {}
        };

        NonbondedMethod nonbondedMethod;
        bool exceptionsUsePeriodic, includeConservative;
        mutable int numContexts;
        double defaultA, defaultGamma, defaultRCut, temperature,
            nonbondedCutoff;
        std::vector<int> particleTypes;
        std::vector<TypePairInfo> typePairs;
        std::vector<ExceptionInfo> exceptions;
        std::unordered_map<std::pair<int, int>, int, PairHash> typePairMap,
            exceptionMap;

        void addExclusionsToSet(const std::vector<std::set<int>> &bonded12,
                                std::set<int> &exclusions, int baseParticle,
                                int fromParticle, int currentLevel) const;
    };

}  // namespace OpenMM

#endif  // OPENMM_DPDFORCE_H_