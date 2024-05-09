#include <string>

#include "ICKernels.h"
#include "ICLangevinIntegrator.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace ICPlugin;
using namespace OpenMM;
using std::string;
using std::vector;

ICLangevinIntegrator::ICLangevinIntegrator(double temperature,
                                           double frictionCoeff,
                                           double stepSize, int numCells,
                                           double cellZSize) {
    setTemperature(temperature);
    setFriction(frictionCoeff);
    setStepSize(stepSize);
    setNumCells(numCells);
    setCellZSize(cellZSize);
    setConstraintTolerance(1e-5);
    setRandomNumberSeed(0);
}

void ICLangevinIntegrator::initialize(ContextImpl& contextRef) {
    if (owner != NULL && &contextRef.getOwner() != owner)
        throw OpenMMException("This Integrator is already bound to a context");
    context = &contextRef;
    owner = &contextRef.getOwner();
    kernel = context->getPlatform().createKernel(
        IntegrateICLangevinStepKernel::Name(), contextRef);
    kernel.getAs<IntegrateICLangevinStepKernel>().initialize(
        contextRef.getSystem(), *this);
}

vector<string> ICLangevinIntegrator::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(IntegrateICLangevinStepKernel::Name());
    return names;
}

double ICLangevinIntegrator::computeKineticEnergy() {
    return kernel.getAs<IntegrateICLangevinStepKernel>().computeKineticEnergy(
        *context, *this);
}

void ICLangevinIntegrator::step(int steps) {
    if (context == NULL)
        throw OpenMMException("This Integrator is not bound to a context!");
    for (int i = 0; i < steps; ++i) {
        context->updateContextState();
        context->calcForcesAndEnergy(true, false);
        kernel.getAs<IntegrateICLangevinStepKernel>().execute(*context, *this);
    }
}