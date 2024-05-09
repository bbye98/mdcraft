#include <sstream>

#include "ICLangevinIntegrator.h"
#include "ICLangevinIntegratorProxy.h"
#include "openmm/serialization/SerializationNode.h"

using namespace std;
using namespace ICPlugin;
using namespace OpenMM;

ICLangevinIntegratorProxy::ICLangevinIntegratorProxy()
    : SerializationProxy("ICLangevinIntegrator") {}

void ICLangevinIntegratorProxy::serialize(const void* object,
                                          SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const ICLangevinIntegrator& integrator =
        *reinterpret_cast<const ICLangevinIntegrator*>(object);
    node.setDoubleProperty("stepSize", integrator.getStepSize());
    node.setDoubleProperty("constraintTolerance",
                           integrator.getConstraintTolerance());
    node.setDoubleProperty("temperature", integrator.getTemperature());
    node.setDoubleProperty("friction", integrator.getFriction());
    node.setIntProperty("randomSeed", integrator.getRandomNumberSeed());
}

void* ICLangevinIntegratorProxy::deserialize(
    const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    ICLangevinIntegrator* integrator = new ICLangevinIntegrator(
        node.getDoubleProperty("temperature"),
        node.getDoubleProperty("friction"), node.getDoubleProperty("stepSize"));
    integrator->setConstraintTolerance(
        node.getDoubleProperty("constraintTolerance"));
    integrator->setRandomNumberSeed(node.getIntProperty("randomSeed"));
    return integrator;
}