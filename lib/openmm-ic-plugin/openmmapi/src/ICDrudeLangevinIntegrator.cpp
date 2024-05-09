#include <string>

#include "ICDrudeLangevinIntegrator.h"
#include "ICKernels.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace ICPlugin;
using namespace OpenMM;
using std::string;
using std::vector;

ICDrudeLangevinIntegrator::ICDrudeLangevinIntegrator(
    double temperature, double frictionCoeff, double drudeTemperature,
    double drudeFrictionCoeff, double stepSize, int numCells, double cellZSize) {
    setTemperature(temperature);
    setFriction(frictionCoeff);
    setDrudeTemperature(drudeTemperature);
    setDrudeFriction(drudeFrictionCoeff);
    setMaxDrudeDistance(0);
    setStepSize(stepSize);
    setNumCells(numCells);
    setCellZSize(cellZSize);
    setConstraintTolerance(1e-5);
    setRandomNumberSeed(0);
}

void ICDrudeLangevinIntegrator::initialize(ContextImpl& contextRef) {
    if (owner != NULL && &contextRef.getOwner() != owner)
        throw OpenMMException("This Integrator is already bound to a context");
    const DrudeForce* force = NULL;
    const System& system = contextRef.getSystem();
    for (int i = 0; i < system.getNumForces(); i++)
        if (dynamic_cast<const DrudeForce*>(&system.getForce(i)) != NULL) {
            if (force == NULL)
                force = dynamic_cast<const DrudeForce*>(&system.getForce(i));
            else
                throw OpenMMException(
                    "The System contains multiple DrudeForces");
        }
    if (force == NULL)
        throw OpenMMException("The System does not contain a DrudeForce");
    context = &contextRef;
    owner = &contextRef.getOwner();
    kernel = context->getPlatform().createKernel(
        IntegrateICDrudeLangevinStepKernel::Name(), contextRef);
    kernel.getAs<IntegrateICDrudeLangevinStepKernel>().initialize(
        contextRef.getSystem(), *this, *force);
}

vector<string> ICDrudeLangevinIntegrator::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(IntegrateICDrudeLangevinStepKernel::Name());
    return names;
}

double ICDrudeLangevinIntegrator::computeKineticEnergy() {
    return kernel.getAs<IntegrateICDrudeLangevinStepKernel>()
        .computeKineticEnergy(*context, *this);
}

void ICDrudeLangevinIntegrator::step(int steps) {
    if (context == NULL)
        throw OpenMMException("This Integrator is not bound to a context!");
    for (int i = 0; i < steps; ++i) {
        context->updateContextState();
        context->calcForcesAndEnergy(true, false);
        kernel.getAs<IntegrateICDrudeLangevinStepKernel>().execute(*context,
                                                                   *this);
    }
}