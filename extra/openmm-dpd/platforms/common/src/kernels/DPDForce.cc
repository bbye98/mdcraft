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
                          int type1, int type2, real3 dr, real4 vel1,
                          real4 vel2, float kBT, mixed dt,
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
    int numAtoms, GLOBAL const real4* RESTRICT posq,
    GLOBAL const real4* RESTRICT velm,
    GLOBAL const int* RESTRICT particleTypeIndices, int numTypes,
    GLOBAL const real4* RESTRICT pairParams, int numExceptions,
    GLOBAL const int2* RESTRICT exceptionParticlePairs,
    GLOBAL const real4* RESTRICT exceptionParams, float kBT, mixed dt,
    mm_long seed, GLOBAL const int2* RESTRICT exceptionTiles,
    int numExceptionTiles, GLOBAL const int* RESTRICT tiles,
    GLOBAL const unsigned int* RESTRICT interactionCount,
    GLOBAL const real4* RESTRICT blockCenter,
    GLOBAL const real4* RESTRICT blockSize,
    GLOBAL const int* RESTRICT interactingAtoms,
    GLOBAL int* RESTRICT tileCounter
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
    mixed halfdt = 0.5f * dt[0].y;
    LOCAL real3 localPos[WORK_GROUP_SIZE];
    LOCAL real4 localVel[WORK_GROUP_SIZE];
    LOCAL int localType[WORK_GROUP_SIZE];
    mixed energy = 0;

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

    // First loop: process fixed tiles (ones that contain exceptions).

    for (int tile = warp; tile < numExceptionTiles; tile += totalWarps) {
        const int2 tileIndices = exceptionTiles[tile];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        int atom1 = x * TILE_SIZE + tgx;
        int type1 = particleTypeIndices[atom1];
        real3 pos1 = trimTo3(posq[atom1]);
        real4 vel1 = velm[atom1];
        if (vel1.w != 0)
            pos1 += halfdt * trimTo3(vel1);
        real3 force1 = make_real3(0);
        if (x == y) {
            localType[LOCAL_ID] = type1;
            localPos[LOCAL_ID] = pos1;
            localVel[LOCAL_ID] = vel1;
        } else {
            int atom2 = y * TILE_SIZE + tgx;
            real3 pos2 = trimTo3(posq[atom2]);
            real4 vel2 = velm[atom2];
            if (vel2.w != 0)
                pos2 += halfdt * trimTo3(vel2);
            localType[LOCAL_ID] = particleType[atom2];
            localPos[LOCAL_ID] = pos2;
            localVel[LOCAL_ID] = vel2;
        }
        SYNC_WARPS;
        if (atom1 < NUM_ATOMS) {
            for (int i = 0; i < TILE_SIZE; i++) {
                int atom2 = y * TILE_SIZE + i;
                if ((x != y || atom1 < atom2) && atom2 < NUM_ATOMS) {
                    real3 pos2 = localPos[tbx + i];
                    real3 delta = make_real3(pos2.x - pos1.x, pos2.y - pos1.y,
                                             pos2.z - pos1.z);
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(delta)
#endif
                    real3 force2 = make_real3(0);
                    real4 params = exceptionParams[type1 * numTypes -
                                                   (type1 * (type1 - 1)) / 2 +
                                                   localType[tbx + i] - type1];
                    computeOneIxn(&energy, &force1, &force2, type1,
                                  localType[tbx + i], delta, vel1,
                                  localVel[tbx + i], params.x, params.y,
                                  params.z, kBT, dt[0].y, &random);
                }
            }
        }
        SYNC_WARPS;
    }

    // Second loop: process tiles from the neighbor list.

    unsigned int numTiles = interactionCount[0];
    LOCAL int atomIndices[WORK_GROUP_SIZE];
    LOCAL int nextTile[WORK_GROUP_SIZE / TILE_SIZE];
    for (int tile = warp; tile < numTiles; tile += totalWarps) {
        if (tgx == 0)
            nextTile[tbx / TILE_SIZE] = ATOMIC_ADD(tileCounter, 1);
        SYNC_WARPS;
        int tileIndex = nextTile[tbx / TILE_SIZE];
        int x = tiles[tileIndex];
        real4 blockSizeX = blockSize[x];
        int atom1 = x * TILE_SIZE + tgx;
        int type1 = particleTypeIndices[atom1];
        real3 pos1 = trimTo3(posq[atom1]);
        real4 vel1 = velm[atom1];
        if (vel1.w != 0)
            pos1 += halfdt * trimTo3(vel1);
        real3 force1 = make_real3(0);
        int atom2 = interactingAtoms[tileIndex * TILE_SIZE + tgx];
        atomIndices[LOCAL_ID] = atom2;
        real3 pos2 = trimTo3(posq[atom2]);
        real4 vel2 = velm[atom2];
        if (vel2.w != 0)
            pos2 += halfdt * trimTo3(vel2);
        localType[LOCAL_ID] = particleTypeIndices[atom2];
        localPos[LOCAL_ID] = pos2;
        localVel[LOCAL_ID] = vel2;
#ifdef USE_PERIODIC
        bool singlePeriodicCopy =
            (0.5f * periodicBoxSize.x - blockSizeX.x >= MAX_CUTOFF &&
             0.5f * periodicBoxSize.y - blockSizeX.y >= MAX_CUTOFF &&
             0.5f * periodicBoxSize.z - blockSizeX.z >= MAX_CUTOFF);
        if (singlePeriodicCopy) {
            // The box is small enough that we can just translate all the atoms
            // into a single periodic box, then skip having to apply periodic
            // boundary conditions later.

            real4 blockCenterX = blockCenter[x];
            APPLY_PERIODIC_TO_POS_WITH_CENTER(pos1, blockCenterX)
            APPLY_PERIODIC_TO_POS_WITH_CENTER(localPos[LOCAL_ID], blockCenterX)
            SYNC_WARPS;
            if (atom1 < NUM_ATOMS) {
                for (int i = 0; i < TILE_SIZE; i++) {
                    int atom2 = atomIndices[tbx + i];
                    if (atom2 < NUM_ATOMS) {
                        pos2 = localPos[tbx + i];
                        real3 delta = make_real3(
                            pos2.x - pos1.x, pos2.y - pos1.y, pos2.z - pos1.z);
                        real3 force2 = make_real3(0);
                        real4 params =
                            exceptionParams[type1 * numTypes -
                                            (type1 * (type1 - 1)) / 2 +
                                            localType[tbx + i] - type1];
                        computeOneIxn(&energy, &force1, &force2, type1,
                                      localType[tbx + i], delta, vel1,
                                      localVel[tbx + i], params.x, params.y,
                                      params.z, kBT, dt[0].y, &random);
                    }
                }
            }
        } else
#endif
        {
            // We need to apply periodic boundary conditions separately for each
            // interaction.

            SYNC_WARPS;
            if (atom1 < NUM_ATOMS) {
                for (int i = 0; i < TILE_SIZE; i++) {
                    int atom2 = atomIndices[tbx + i];
                    if (atom2 < NUM_ATOMS) {
                        pos2 = localPos[tbx + i];
                        real3 delta = make_real3(
                            pos2.x - pos1.x, pos2.y - pos1.y, pos2.z - pos1.z);
#ifdef USE_PERIODIC
                        APPLY_PERIODIC_TO_DELTA(delta)
#endif
                        real3 force2 = make_real3(0);
                        real4 params =
                            exceptionParams[type1 * numTypes -
                                            (type1 * (type1 - 1)) / 2 +
                                            localType[tbx + i] - type1];
                        computeOneIxn(&energy, &force1, &force2, type1,
                                      localType[tbx + i], delta, vel1,
                                      localVel[tbx + i], params.x, params.y,
                                      params.z, kBT, dt[0].y, &random);
                    }
                    ATOMIC_ADD(&forceBuffers[atom2],
                               (mm_ulong)realToFixedPoint(force2.x));
                    ATOMIC_ADD(&forceBuffers[atom2 + PADDED_NUM_ATOMS],
                               (mm_ulong)realToFixedPoint(force2.y));
                    ATOMIC_ADD(&forceBuffers[atom2 + 2 * PADDED_NUM_ATOMS],
                               (mm_ulong)realToFixedPoint(force2.z));
                }
            }
        }
        ATOMIC_ADD(&forceBuffers[atom1], (mm_ulong)realToFixedPoint(force1.x));
        ATOMIC_ADD(&forceBuffers[atom1 + PADDED_NUM_ATOMS],
                   (mm_ulong)realToFixedPoint(force1.y));
        ATOMIC_ADD(&forceBuffers[atom1 + 2 * PADDED_NUM_ATOMS],
                   (mm_ulong)realToFixedPoint(force1.z));
        SYNC_WARPS;
    }
    energyBuffer[GLOBAL_ID] += energy;
}