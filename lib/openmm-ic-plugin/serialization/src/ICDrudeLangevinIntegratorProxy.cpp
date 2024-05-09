#include <sstream>

#include "ICDrudeLangevinIntegrator.h"
#include "ICDrudeLangevinIntegratorProxy.h"
#include "openmm/serialization/SerializationNode.h"

using namespace std;
using namespace ICPlugin;
using namespace OpenMM;

ICDrudeLangevinIntegratorProxy::ICDrudeLangevinIntegratorProxy()
    : SerializationProxy("ICDrudeLangevinIntegrator") {}

void ICDrudeLangevinIntegratorProxy::serialize(const void* object,
                                               SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const ICDrudeLangevinIntegrator& integrator =
        *reinterpret_cast<const ICDrudeLangevinIntegrator*>(object);
    node.setDoubleProperty("stepSize", integrator.getStepSize());
    node.setDoubleProperty("constraintTolerance",
                           integrator.getConstraintTolerance());
    node.setDoubleProperty("temperature", integrator.getTemperature());
    node.setDoubleProperty("friction", integrator.getFriction());
    node.setDoubleProperty("drudeTemperature",
                           integrator.getDrudeTemperature());
    node.setDoubleProperty("drudeFriction", integrator.getDrudeFriction());
    node.setIntProperty("randomSeed", integrator.getRandomNumberSeed());
}

void* ICDrudeLangevinIntegratorProxy::deserialize(
    const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    ICDrudeLangevinIntegrator* integrator = new ICDrudeLangevinIntegrator(
        node.getDoubleProperty("temperature"),
        node.getDoubleProperty("friction"),
        node.getDoubleProperty("drudeTemperature"),
        node.getDoubleProperty("drudeFriction"),
        node.getDoubleProperty("stepSize"));
    integrator->setConstraintTolerance(
        node.getDoubleProperty("constraintTolerance"));
    integrator->setRandomNumberSeed(node.getIntProperty("randomSeed"));
    return integrator;
}