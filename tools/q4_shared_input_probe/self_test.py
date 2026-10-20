import json
import struct

import generate
import layout_proof
import sass_gate


def require_reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid evidence was accepted")


def main():
    symbol = "_ZN6ninfer3ops4gguf6detail3vec6kernelIL9ggml_type12ELi4ELb1ELi5120ELb1ELb0ELi2EEEvNS2_7VecArgsE"
    resource = sass_gate.resources("Function " + symbol + ":\n REG:99 STACK:0 SHARED:0 LOCAL:0\n", symbol)
    stack = sass_gate.ptxas_stack("ptxas info : Function properties for " + symbol +
                                "\n 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n", symbol)
    assert sass_gate.resource_bound(resource, stack, 23040)["minimum_capacity"] == 4
    assert sass_gate.resource_bound(resource, stack, 25600)["status"] == "FAIL_FOUR_CTA_RESOURCE_BOUND"
    assert sass_gate.resource_bound(dict(resource, REG=129), stack, 23040)["status"].startswith("FAIL")
    assert sass_gate.resource_bound(resource, dict(stack, spill_load_bytes=4), 23040)["status"].startswith("FAIL")
    require_reject(lambda: sass_gate.resources("Function " + symbol + ":\n REG:99 STACK:0 SHARED:0\n", symbol))
    require_reject(lambda: sass_gate.ptxas_stack("Used 99 registers", symbol))
    sass = ("Function : " + symbol + "\n"
            " /*0000*/ LDG.E.128 R8, [R2];\n"
            " /*0010*/ IDP.4A.S8.S8 R9, R10, R11, R9;\n"
            " /*0020*/ SHFL.BFLY PT, R9, R9, 0x10, 0x1f;\n"
            " /*0030*/ @P0 BRA 0x10;\n")
    _, old = sass_gate.selected_sass(sass)
    _, bad = sass_gate.selected_sass(sass.replace(" /*0020*/", " /*0018*/ BAR.SYNC 0x0;\n /*0020*/"))
    _, staged = sass_gate.selected_sass(sass.replace(" /*0020*/", " /*0018*/ LDS.128 R12, [R14];\n /*0020*/"))
    assert sass_gate.structural_review(old, bad)["status"] == "FAIL_NEW_CTA_BARRIER_IN_ROW_LOOP"
    assert sass_gate.structural_review(old, old)["status"] == "FAIL_NO_SHARED_LOAD_IN_ROW_LOOP"
    assert sass_gate.structural_review(old, staged)["status"] == "MANUAL_REQUIRED"
    assert sass_gate.structural_review(old, staged)["operand_aware_math_DAG_proven"] is False
    names = b"\0.shstrtab\0.text." + symbol.encode() + b"\0"
    text = bytes(range(32))
    names_offset = 64
    code_offset = names_offset + len(names)
    sections_offset = code_offset + len(text)
    header = struct.pack("<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01" + b"\0" * 10,
                         2, 190, 1, 0, 0, sections_offset, 0, 64, 0, 0, 64, 3, 1)
    sections = (bytes(64) + struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, names_offset, len(names), 0, 0, 1, 0)
                + struct.pack("<IIQQQQIIQQ", 11, 1, 4, 0, code_offset, len(text), 0, 0, 16, 0))
    elf = header + names + text + sections
    assert sass_gate.elf_text(elf, symbol) == text
    require_reject(lambda: sass_gate.elf_text(elf[:-1], symbol))
    require_reject(lambda: sass_gate.elf_text(elf, symbol + "_wrong"))
    outputs, receipt = generate.prepare(generate.ROOT)
    assert receipt["OFF_native_byte_identity_after_include_rebinding"]
    assert receipt["ON_reverse_transform_byte_exact"]
    assert outputs["q4_shared_input_generated.cuh"].count(b"resident_codes[i ^ ((i >> 3) & 1)]") == 1
    print(json.dumps({"status": "PASS_SOURCE_AND_FAIL_CLOSED_PARSER_SELF_TESTS",
                      "layout": layout_proof.prove(), "CPU_checks_only": True,
                      "compile_run": False, "GPU_run": False}, indent=2))


if __name__ == "__main__":
    main()
