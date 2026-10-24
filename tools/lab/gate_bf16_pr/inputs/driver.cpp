#include <cstring>
#include <iostream>
int ninfer_bundle_entry_lab_swiglu_bench(int, char**);
int ninfer_bundle_entry_ninfer_linear_gguf_test();
int ninfer_bundle_entry_ninfer_linear_gguf_mmq_store_test();
int ninfer_bundle_entry_ninfer_linear_bench(int, char**);
int main(int argc, char** argv) {
    if (argc>1 && std::strcmp(argv[1],"swiglu")==0) { return ninfer_bundle_entry_lab_swiglu_bench(argc-1,argv+1); }
    if (argc>1 && std::strcmp(argv[1],"oracle")==0) { return ninfer_bundle_entry_ninfer_linear_gguf_test(); }
    if (argc>1 && std::strcmp(argv[1],"store")==0) { return ninfer_bundle_entry_ninfer_linear_gguf_mmq_store_test(); }
    if (argc>1 && std::strcmp(argv[1],"bench")==0) { return ninfer_bundle_entry_ninfer_linear_bench(argc-1,argv+1); }
    std::cerr << "select swiglu, oracle, store or bench\n";
    return 2;
}
