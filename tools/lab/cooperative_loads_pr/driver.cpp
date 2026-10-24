#include <cstring>
#include <iostream>

int ninfer_bundle_entry_ninfer_linear_gguf_test();
int ninfer_bundle_entry_ninfer_linear_bench(int, char**);
#if PR_HAS_STORE
int ninfer_bundle_entry_ninfer_linear_gguf_mmq_store_test();
#endif

#if PR_HAS_COOP
int ninfer_bundle_entry_ninfer_linear_gguf_cooperative_test();
#endif

int main(int argc, char** argv) {
#if PR_HAS_COOP
    if (argc > 1 && std::strcmp(argv[1], "cooperative") == 0) {
        return ninfer_bundle_entry_ninfer_linear_gguf_cooperative_test();
    }
#endif
    if (argc > 1 && std::strcmp(argv[1], "oracle") == 0) {
        return ninfer_bundle_entry_ninfer_linear_gguf_test();
    }
    if (argc > 1 && std::strcmp(argv[1], "bench") == 0) {
        return ninfer_bundle_entry_ninfer_linear_bench(argc - 1, argv + 1);
    }
#if PR_HAS_STORE
    if (argc > 1 && std::strcmp(argv[1], "store") == 0) {
        return ninfer_bundle_entry_ninfer_linear_gguf_mmq_store_test();
    }
#endif
    std::cerr << "select oracle, bench or store\n";
    return 2;
}
