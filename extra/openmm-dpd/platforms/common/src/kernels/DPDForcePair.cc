// #ifdef USE_CUTOFF
// if (!isExcluded && r2 < CUTOFF_SQUARED) {
// #else
// if (!isExcluded) {
// #endif
//     real tempForce = 0;
//     // COMPUTE_FORCE
//     tempEnergy += customEnergy;
// #endif
//     dEdR += tempForce * invR;
// }

#ifdef USE_CUTOFF
if (!isExcluded && r2 < CUTOFF_SQUARED) {
#else
if (!isExcluded) {
#endif
    real ijForce = 0;
}