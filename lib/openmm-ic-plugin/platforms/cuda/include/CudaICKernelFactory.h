#ifndef OPENMM_CUDAICKERNELFACTORY_H_
#define OPENMM_CUDAICKERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

class CudaICKernelFactory : public KernelFactory {
   public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform,
                                 ContextImpl& context) const;
};

}  // namespace OpenMM

#endif /*OPENMM_CUDAICKERNELFACTORY_H_*/