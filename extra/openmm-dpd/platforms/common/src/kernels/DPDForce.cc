KERNEL void computeIxns(GLOBAL mixed* RESTRICT energyBuffer,
                        GLOBAL mm_ulong* RESTRICT forceBuffers, int numAtoms,
                        int numExceptions,
                        GLOBAL const real4* RESTRICT positions,
                        GLOBAL const int* RESTRICT particleTypeIndices,
                        GLOBAL const real4* RESTRICT pairParams,
                        GLOBAL const int2* RESTRICT exceptionParticlePairs,
                        GLOBAL const real4* RESTRICT exceptionParams) {}

DEVICE void computeOneIxn(mixed* totalEnergy, real3* force1, real3* force2,
                          real3 pos1, real3 pos2, real3 vel1, real3 vel2,
                          float A, float gamma, float rCut, float dt,
                          RandomState* random
#ifdef USE_PERIODIC
                          ,
                          real4 periodicBoxSize, real4 invPeriodicBoxSize,
                          real4 periodicBoxVecX, real4 periodicBoxVecY,
                          real4 periodicBoxVecZ
#endif
) {
    real3 dr = pos1 - pos2;
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(dr)
#endif
    real r2 = dot(dr, dr);
}