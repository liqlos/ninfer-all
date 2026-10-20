#ifdef NINFER_Q6_FULL_WARP_SLICES
#error "the variant translation unit owns the FullWarpSlices switch"
#endif
#define NINFER_Q6_FULL_WARP_SLICES 1
#define NINFER_Q6_FULL_WARP_PROBE_ENTRY q6_full_warp_probe_on
#include "variant_impl.cuh"
