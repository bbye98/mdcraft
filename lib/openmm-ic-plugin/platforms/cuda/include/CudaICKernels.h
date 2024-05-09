#ifndef CUDA_IC_KERNELS_H_
#define CUDA_IC_KERNELS_H_

#include "CudaArray.h"
#include "CudaContext.h"
#include "ICKernels.h"

using namespace ICPlugin;

namespace OpenMM {

class CudaIntegrateICLangevinStepKernel : public IntegrateICLangevinStepKernel {
   public:
    CudaIntegrateICLangevinStepKernel(std::string name,
                                      const Platform& platform, CudaContext& cu)
        : IntegrateICLangevinStepKernel(name, platform),
          cu(cu),
          params(NULL),
          invAtomIndex(NULL) {}
    ~CudaIntegrateICLangevinStepKernel();

    /**
     * Initialize the kernel, setting up the particle masses.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the ICLangevinIntegrator this kernel will be used for
     */
    void initialize(const System& system,
                    const ICLangevinIntegrator& integrator);

    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the ICLangevinIntegrator this kernel is being used for
     */
    void execute(ContextImpl& context, const ICLangevinIntegrator& integrator);

    /**
     * Compute the kinetic energy.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the ICLangevinIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context,
                                const ICLangevinIntegrator& integrator);

   private:
    CudaContext& cu;
    double prevTemp, prevFriction, prevStepSize, cellZSize;
    CudaArray *params, *invAtomIndex;
    CUfunction kernel1, kernel2, kernelImage, kernelReorder;
};

class CudaIntegrateICDrudeLangevinStepKernel
    : public IntegrateICDrudeLangevinStepKernel {
   public:
    CudaIntegrateICDrudeLangevinStepKernel(std::string name,
                                           const Platform& platform,
                                           CudaContext& cu)
        : IntegrateICDrudeLangevinStepKernel(name, platform), cu(cu) {}

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the ICDrudeLangevinIntegrator this kernel will be used
     * for
     * @param force      the DrudeForce to get particle parameters from
     */
    void initialize(const System& system,
                    const ICDrudeLangevinIntegrator& integrator,
                    const DrudeForce& force);

    /**
     * Execute the kernel.
     *
     * @param context        the context in which to execute this kernel
     * @param integrator     the ICDrudeLangevinIntegrator this kernel is being
     * used for
     */
    void execute(ContextImpl& context,
                 const ICDrudeLangevinIntegrator& integrator);

    /**
     * Compute the kinetic energy.
     *
     * @param context     the context in which to execute this kernel
     * @param integrator  the ICDrudeLangevinIntegrator this kernel is being
     * used for
     */
    double computeKineticEnergy(ContextImpl& context,
                                const ICDrudeLangevinIntegrator& integrator);

   private:
    CudaContext& cu;
    double prevStepSize, cellZSize;
    CudaArray normalParticles, pairParticles, invAtomIndex;
    CUfunction kernel1, kernel2, hardwallKernel, kernelImage, kernelReorder;
};

}  // namespace OpenMM

#endif /*CUDA_IC_KERNELS_H_*/