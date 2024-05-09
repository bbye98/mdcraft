#include <exception>

#include "CudaICKernelFactory.h"
#include "CudaICKernels.h"
#include "internal/windowsExportIC.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;

extern "C" OPENMM_EXPORT_IC void registerPlatforms() {
}

extern "C" OPENMM_EXPORT_IC void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaICKernelFactory* factory = new CudaICKernelFactory();
        platform.registerKernelFactory(IntegrateICLangevinStepKernel::Name(), factory);
        platform.registerKernelFactory(IntegrateICDrudeLangevinStepKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT_IC void registerCudaICKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaICKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == IntegrateICLangevinStepKernel::Name())
        return new CudaIntegrateICLangevinStepKernel(name, platform, cu);
    if (name == IntegrateICDrudeLangevinStepKernel::Name())
        return new CudaIntegrateICDrudeLangevinStepKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}