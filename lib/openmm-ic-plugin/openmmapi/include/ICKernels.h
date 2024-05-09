#ifndef IC_KERNELS_H_
#define IC_KERNELS_H_

#include <string>
#include <vector>

#include "ICDrudeLangevinIntegrator.h"
#include "ICLangevinIntegrator.h"
#include "openmm/DrudeForce.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"

namespace ICPlugin {

/**
 * This kernel is invoked by ICLangevinIntegrator to take one time step.
 */
class IntegrateICLangevinStepKernel : public OpenMM::KernelImpl {
   public:
    static std::string Name() { return "IntegrateICLangevinStep"; }
    IntegrateICLangevinStepKernel(std::string name,
                                  const OpenMM::Platform& platform)
        : KernelImpl(name, platform) {}

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the ICLangevinIntegrator this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system,
                            const ICLangevinIntegrator& integrator) = 0;

    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the ICLangevinIntegrator this kernel is being used for
     */
    virtual void execute(OpenMM::ContextImpl& context,
                         const ICLangevinIntegrator& integrator) = 0;

    /**
     * Compute the kinetic energy.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the ICLangevinIntegrator this kernel is being used for
     */
    virtual double computeKineticEnergy(
        OpenMM::ContextImpl& context,
        const ICLangevinIntegrator& integrator) = 0;
};

/**
 * This kernel is invoked by ICDrudeLangevinIntegrator to take one time step.
 */
class IntegrateICDrudeLangevinStepKernel : public OpenMM::KernelImpl {
   public:
    static std::string Name() { return "IntegrateICDrudeLangevinStep"; }
    IntegrateICDrudeLangevinStepKernel(std::string name,
                                       const OpenMM::Platform& platform)
        : KernelImpl(name, platform) {}

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the ICDrudeLangevinIntegrator this kernel will be used
     * for
     * @param force      the DrudeForce to get particle parameters from
     */
    virtual void initialize(const OpenMM::System& system,
                            const ICDrudeLangevinIntegrator& integrator,
                            const OpenMM::DrudeForce& force) = 0;

    /**
     * Execute the kernel.
     *
     * @param context        the context in which to execute this kernel
     * @param integrator     the ICDrudeLangevinIntegrator this kernel is being
     * used for
     */
    virtual void execute(OpenMM::ContextImpl& context,
                         const ICDrudeLangevinIntegrator& integrator) = 0;

    /**
     * Compute the kinetic energy.
     */
    virtual double computeKineticEnergy(
        OpenMM::ContextImpl& context,
        const ICDrudeLangevinIntegrator& integrator) = 0;
};

}  // namespace ICPlugin

#endif /*IC_KERNELS_H_*/