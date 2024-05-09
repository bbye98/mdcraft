#ifdef WIN32
#include <windows.h>

#include <sstream>
#else
#include <dirent.h>
#include <dlfcn.h>

#include <cstdlib>
#endif

#include "ICDrudeLangevinIntegrator.h"
#include "ICDrudeLangevinIntegratorProxy.h"
#include "ICLangevinIntegrator.h"
#include "ICLangevinIntegratorProxy.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationProxy.h"

#if defined(WIN32)
#include <windows.h>
extern "C" void registerICSerializationProxies();
BOOL WINAPI DllMain(HANDLE hModule, DWORD ul_reason_for_call,
                    LPVOID lpReserved) {
    if (ul_reason_for_call == DLL_PROCESS_ATTACH)
        registerICSerializationProxies();
    return TRUE;
}
#else
extern "C" void __attribute__((constructor)) registerICSerializationProxies();
#endif

using namespace OpenMM;

extern "C" void registerICSerializationProxies() {
    SerializationProxy::registerProxy(typeid(ICPlugin::ICLangevinIntegrator),
                                      new ICLangevinIntegratorProxy());
    SerializationProxy::registerProxy(
        typeid(ICPlugin::ICDrudeLangevinIntegrator),
        new ICDrudeLangevinIntegratorProxy());
}