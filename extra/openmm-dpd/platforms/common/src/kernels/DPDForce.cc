static constexpr double EPSILON = 1.0e-12;

DEVICE unsigned int getRandomInt(RandomState* random) {
    unsigned int xs = ((random->state >> 18) ^ random->state) >> 27;
    unsigned int rot = random->state >> 59;
    random->state = random->state * 6364136223846793005ULL + random->increment;
    return (xs >> rot) | (xs << ((-rot) & 31));
}

DEVICE float getRandomNormal(RandomState* random) {
    if (random->nextIsValid) {
        random->nextIsValid = false;
        return random->next;
    }
    float scale = 1 / (float)0x100000000;
    float x = scale * max(getRandomInt(random), 1u);
    float y = scale * getRandomInt(random);
    float multiplier = SQRT(-2 * LOG(x));
    float angle = 2 * M_PI * y;
    random->next = multiplier * COS(angle);
    random->nextIsValid = true;
    return multiplier * SIN(angle);
}

DEVICE void computeOneIxn(mixed* totalEnergy, real3* force1, real3* force2,
                          real3 dr, real3 vel1, real3 vel2, float A,
                          float gamma, float rCut, float kBT, mixed dt,
                          RandomState* random) {
    real r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
    real invR = RSQRT(r2);
    real r = r2 * invR;
    if (r < rCut) {
        if (r < EPSILON) {
#ifdef INCLUDE_CONSERVATIVE
            *totalEnergy += 0.5f * A;
#endif
            return;
        }

        real weight = 1.0 - r / rCut;
        real weight2 = weight * weight;
        real3 drUnitVector = dr * invR;
        real3 dv =
            make_real3(vel2.x - vel1.x, vel2.y - vel1.y, vel2.z - vel1.z);

        real forceMag =
#ifdef INCLUDE_CONSERVATIVE
            A * weight
#endif
            - gamma * weight2 * dot(drUnitVector, dv) +
            SQRT(2 * gamma * kBT / dt) * weight * getRandomNormal(random);
        real3 force = drUnitVector * forceMag;
        *force1 += force;
        *force2 -= force;
        *totalEnergy += 0.5f * A * weight2;
    }
}

KERNEL void computeIxns(
    GLOBAL mixed* RESTRICT energyBuffer, GLOBAL mm_ulong* RESTRICT forceBuffers,
    int numAtoms, GLOBAL const real4* RESTRICT positions,
    GLOBAL const int* RESTRICT particleTypeIndices,
    GLOBAL const real4* RESTRICT pairParams, int numExceptions,
    GLOBAL const int2* RESTRICT exceptionParticlePairs,
    GLOBAL const real4* RESTRICT exceptionParams, float kBT, mixed dt
#ifdef USE_PERIODIC
    ,
    real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX,
    real4 periodicBoxVecY, real4 periodicBoxVecZ
#endif
) {

    const unsigned int totalWarps = GLOBAL_SIZE / TILE_SIZE;
    const unsigned int warp = GLOBAL_ID / TILE_SIZE;  // global warpIndex
    const unsigned int tgx =
        LOCAL_ID & (TILE_SIZE - 1);           // index within the warp
    const unsigned int tbx = LOCAL_ID - tgx;  // block warpIndex

    // Initialize the random generator for this thread. The seed is incremented
    // each step, and the stream ID is the global thread index. Skipping a
    // variable number of values also seems to be necessary to decorrelate the
    // streams.

    RandomState random;
    random.state = 0;
    random.increment = (GLOBAL_ID << 1) | 1;
    random.nextIsValid = false;
    getRandomInt(&random);
    random.state += seed;
    getRandomInt(&random);
    for (int i = 0; i < LOCAL_ID % 16; i++) getRandomInt(&random);
}