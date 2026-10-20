import argparse
import collections
import hashlib
import json
import pathlib
import re
import sys


SIGNATURE = re.compile(
    r"vec6kernelIL9ggml_type14ELi([14])ELb0ELi0ELb1ELb0ELi2ELb([01])EEEv")
INSTRUCTION = re.compile(
    r"/\*(?P<pc>[0-9a-fA-F]+)\*/\s*(?P<predicate>@!?(?:UP|P)(?:T|\d+)\s+)?"
    r"(?P<opcode>[A-Z][A-Z0-9_.]*)\s*(?P<operands>[^;]*);")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(text):
    matches = list(re.finditer(r"^\s*Function\s*:\s*(\S+)\s*$", text, re.MULTILINE))
    result = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        name = match.group(1)
        if name in result:
            raise ValueError("duplicate SASS function; require one SM86 image")
        result[name] = text[match.start():end]
    return result


def instructions(body):
    result = []
    for match in INSTRUCTION.finditer(body):
        result.append((int(match.group("pc"), 16), match.group("opcode"), match.group("operands").strip()))
    if not result or len({pc for pc, _, _ in result}) != len(result):
        raise ValueError("missing instructions or duplicate PCs in selected function")
    return result


def histogram(items):
    return dict(sorted(collections.Counter(op for _, op, _ in items).items()))


def mask_chains(items):
    votes, uniform, chains = {}, {}, []
    for pc, op, operands in items:
        words = [part.strip().split(".")[0] for part in operands.split(",")]
        if op == "VOTE.ANY" and re.fullmatch(r"R\d+", words[0]):
            votes[words[0]] = pc
        elif op == "R2UR" and len(words) == 2:
            uniform.pop(words[0], None)
            if words[1] in votes:
                uniform[words[0]] = (votes[words[1]], pc)
        elif op == "BRA.DIV":
            register = words[0].lstrip("~")
            if register in uniform:
                vote, transfer = uniform[register]
                chains.append({"vote": hex(vote), "r2ur": hex(transfer), "bra_div": hex(pc)})
        else:
            destination = words[1] if op.startswith("SHFL.") and len(words) > 1 else words[0]
            if not op.startswith(("ST", "BRA", "CALL", "ISETP", "FSETP", "PLOP", "BSSY")):
                votes.pop(destination, None)
                uniform.pop(destination, None)
    return chains


def runtime_loop(items, columns):
    loops = []
    for end, op, operands in items:
        if op != "BRA":
            continue
        target = re.fullmatch(r"0x([0-9a-fA-F]+)", operands)
        if target is None:
            continue
        begin = int(target.group(1), 16)
        if begin >= end:
            continue
        region = [item for item in items if begin <= item[0] <= end]
        hist = histogram(region)
        if hist.get("IDP.4A.S8.S8", 0) == 16 * columns and hist.get("SHFL.IDX", 0) == 32:
            loops.append((begin, end, region))
    minimal = [loop for loop in loops if not any(
        other[0] >= loop[0] and other[1] <= loop[1] and other[:2] != loop[:2] for other in loops)]
    if len(minimal) != 1:
        raise ValueError("cannot uniquely identify the non-unrolled native K loop; inspect selected SASS")
    begin, end, region = minimal[0]
    return {"begin": hex(begin), "backedge": hex(end), "opcodes": histogram(region),
            "mask_chains": mask_chains(region)}, region


def divergence_guards(body, items, loop):
    predicates = {int(match.group("pc"), 16): (match.group("predicate") or "PT").strip()
                  for match in INSTRUCTION.finditer(body)}
    positions = {pc: index for index, (pc, _, _) in enumerate(items)}
    begin, end = int(loop["begin"], 16), int(loop["backedge"], 16)

    def direct_target(operands):
        match = re.fullmatch(r"0x([0-9a-fA-F]+)", operands)
        return int(match.group(1), 16) if match else None

    def straight_block(target):
        if target not in positions:
            return []
        result = []
        for item in items[positions[target]:]:
            result.append(item)
            if item[1].startswith(("BRA", "RET", "EXIT", "BRX", "JMP", "JMX")):
                break
        return result

    def latest_mask_definition(register, before):
        for pc, op, operands in reversed(items[:positions[before]]):
            words = [word.strip().split(".")[0] for word in operands.split(",")]
            destination = words[1] if op.startswith("SHFL.") and len(words) > 1 else words[0]
            if destination != register or op.startswith(("ST", "BRA", "CALL", "RET", "WARPSYNC")):
                continue
            full = ((op in ("MOV", "IMAD.MOV.U32") and words[-1] in ("-0x1", "0xffffffff"))
                    and predicates[pc] == "PT")
            return {"pc": hex(pc), "opcode": op, "operands": operands,
                    "predicate": predicates[pc], "full_mask_literal": full}
        return None

    result = []
    for pc, op, operands in items:
        if op != "BRA.DIV" or not begin <= pc <= end:
            continue
        match = re.fullmatch(r"([^,]+),\s*0x([0-9a-fA-F]+)", operands)
        operand = match.group(1).strip() if match else None
        target = int(match.group(2), 16) if match else None
        block = straight_block(target)
        helpers = {}
        for call_pc, call_op, call_args in block:
            if not call_op.startswith("CALL."):
                continue
            helper_target = direct_target(call_args)
            if helper_target is None or helper_target in helpers:
                continue
            helper = straight_block(helper_target)
            sync = [(at, args) for at, opcode, args in helper if opcode == "WARPSYNC"]
            helper_hist = histogram(helper)
            helpers[helper_target] = {
                "entry": hex(helper_target), "first_call": hex(call_pc),
                "opcodes": helper_hist,
                "synchronized_idx_shuffle": len(sync) == 1 and
                    helper_hist.get("SHFL.IDX", 0) == 1 and
                    helper_hist.get("RET.REL.NODEC", 0) == 1 and
                    sync[0][0] < next(at for at, opcode, _ in helper if opcode == "SHFL.IDX"),
                "warpsync": [{"pc": hex(at), "mask_register": register,
                               "definition_before_guard": latest_mask_definition(register, pc)}
                              for at, register in sync],
            }
        rejoin = direct_target(block[-1][2]) if block and block[-1][1] == "BRA" else None
        recognized = (operand == "~URZ" and predicates[pc] == "PT" and
                      target is not None and target > end and rejoin is not None and
                      begin <= rejoin <= end and predicates[block[-1][0]] == "PT" and
                      bool(helpers) and all(
                          helper["synchronized_idx_shuffle"] and all(
                              sync["definition_before_guard"] is not None and
                              sync["definition_before_guard"]["full_mask_literal"]
                              for sync in helper["warpsync"])
                          for helper in helpers.values()))
        result.append({
            "pc": hex(pc), "opcode": op, "operands": operands, "predicate": predicates[pc],
            "mask_operand": operand, "target": hex(target) if target is not None else None,
            "source": "constant_full_warp_mask" if operand == "~URZ" else "runtime_or_unrecognized_mask",
            "classification": ("CONSTANT_FULL_MASK_SYNC_GUARD" if recognized
                               else "UNRESOLVED_DIVERGENCE_GUARD"),
            "target_opcodes": histogram(block),
            "rejoin": hex(rejoin) if rejoin is not None else None,
            "synchronizing_helpers": list(helpers.values()),
            "dead_branch_proven": False,
            "taken_frequency_measured": False,
        })
    return result


def resources(text, name):
    match = re.search(r"^\s*Function\s+" + re.escape(name) + r":\s*\n([^\n]+)", text, re.MULTILINE)
    if match is None:
        raise ValueError("selected kernel is absent from actual resource dump")
    result = {key: int(value) for key, value in re.findall(r"([A-Z]+):(\d+)", match.group(1))}
    if not {"REG", "STACK", "SHARED", "LOCAL"} <= result.keys():
        raise ValueError("selected resource dump is incomplete")
    return result


def dependencies(path, source_root):
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    _, separator, prerequisites = text.partition(":")
    if not separator:
        raise ValueError("invalid compiler dependency output")
    return {(source_root / item).resolve() for item in prerequisites.split()}


def run(root, read_only=False):
    metadata = json.loads((root / "generated.json").read_text(encoding="utf-8"))
    if metadata["status"] != "PASS_BOUNDED_FULL_WARP_SOURCE_TRANSFORM" or metadata["check_only"]:
        raise ValueError("missing completed pinned generation")
    source_root = pathlib.Path(metadata["source_root"])
    probe_root = pathlib.Path(metadata["probe_root"])
    generated_path = pathlib.Path(metadata["generated_path"]).resolve()
    if sha(generated_path) != metadata["generated_header_sha256"]:
        raise ValueError("generated header changed after generation")
    for directory, files in ((source_root, metadata["source_files"]),
                             (probe_root, metadata["probe_files"])):
        for name, expected in files.items():
            if directory == probe_root and name == "sass_gate.py":
                continue
            if sha(directory / name) != expected:
                raise ValueError("pinned source or probe changed after generation: " + name)
    analyzer_path = pathlib.Path(__file__).resolve()
    analyzer = {"current_sha256": sha(analyzer_path),
                "at_generation_sha256": metadata["probe_files"]["sass_gate.py"]}
    analyzer["revised_since_generation"] = analyzer["current_sha256"] != analyzer["at_generation_sha256"]
    actual_dependencies = {}
    all_kernels = {}
    receipts = {}
    for variant, enabled in (("off", "0"), ("on", "1")):
        deps = dependencies(root / (variant + ".d"), source_root)
        required = {generated_path, (probe_root / "variant_impl.cuh").resolve(),
                    (probe_root / ("variant_" + variant + ".cu")).resolve()}
        required.update((source_root / name).resolve() for name in metadata["source_files"]
                        if name not in ("src/ops/linear/gguf/ggml_bridge_vec.cuh", "src/ops/common/mma.cuh"))
        if not required <= deps:
            raise ValueError("actual compiler include set missed pinned inputs: " +
                             ", ".join(str(path) for path in required - deps))
        if (source_root / "src/ops/linear/gguf/ggml_bridge_vec.cuh").resolve() in deps:
            raise ValueError("native vector header leaked into selected TU")
        if analyzer_path in deps or (probe_root / "sass_gate.py").resolve() in deps:
            raise ValueError("analysis script unexpectedly appears among compiled inputs")
        actual_dependencies[variant] = deps - {(probe_root / ("variant_" + variant + ".cu")).resolve()}
        sass_text = (root / ("sass-" + variant + ".txt")).read_text(encoding="utf-8")
        resource_text = (root / ("resources-" + variant + ".txt")).read_text(encoding="utf-8")
        if set(re.findall(r"code for (sm_\d+)", sass_text)) != {"sm_86"}:
            raise ValueError("actual SASS is not a single SM86 target")
        selected = {name: body for name, body in functions(sass_text).items() if "vec6kernelI" in name}
        if len(selected) != 2:
            raise ValueError("expected exactly two selected native vector kernels per TU")
        kernels = {}
        for name, body in selected.items():
            signature = SIGNATURE.search(name)
            if signature is None or signature.group(2) != enabled:
                raise ValueError("actual selected kernel has unexpected template arguments: " + name)
            columns = int(signature.group(1))
            items = instructions(body)
            loop, loop_items = runtime_loop(items, columns)
            loop["divergence_guards"] = divergence_guards(body, items, loop)
            resource = resources(resource_text, name)
            local_ops = [op for _, op, _ in items if op.startswith(("LDL", "STL"))]
            spills = resource["STACK"] != 0 or resource["LOCAL"] != 0 or bool(local_ops)
            body_path = root / ("selected-" + variant + "-t" + str(columns) + ".sass")
            if not read_only:
                body_path.write_text(body, encoding="utf-8")
            kernels[columns] = {"function": name, "resources": resource, "spills_or_local": spills,
                                "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                                "body_opcodes": histogram(items),
                                "runtime_loop": loop}
        if set(kernels) != {1, 4}:
            raise ValueError("missing selected T1 or T4")
        all_kernels[variant] = kernels
        receipts[variant] = {suffix: sha(root / suffix) for suffix in (
            variant + ".o", variant + ".d", "compile-" + variant + ".log",
            "resources-" + variant + ".txt", "sass-" + variant + ".txt")}
    if actual_dependencies["off"] != actual_dependencies["on"]:
        raise ValueError("ON/OFF actual dependency sets differ beyond their selected TU")
    routes = {}
    for columns in (1, 4):
        off, on = (all_kernels[variant][columns] for variant in ("off", "on"))
        a, b = off["runtime_loop"], on["runtime_loop"]
        float_ops = {op for op in a["opcodes"] | b["opcodes"]
                     if op.startswith(("FADD", "FFMA", "FMUL", "I2F", "HADD2", "HFMA2", "HMUL2"))}
        arithmetic_counts_match = all(a["opcodes"].get(op, 0) == b["opcodes"].get(op, 0)
                                      for op in float_ops | {"IDP.4A.S8.S8", "SHFL.IDX"})
        if off["spills_or_local"] or on["spills_or_local"] or not arithmetic_counts_match:
            status = "FAIL_RESOURCE_OR_ARITHMETIC_SHAPE"
        elif not a["mask_chains"] and not any(a["opcodes"].get(op, 0)
                                               for op in ("VOTE.ANY", "R2UR", "BRA.DIV")):
            status = "NO_MECHANISM_BASELINE_ALREADY_UNIFORM"
        elif not a["mask_chains"]:
            status = "UNRESOLVED_BASELINE_MASK_CHAIN"
        elif (b["mask_chains"] or b["opcodes"].get("VOTE.ANY", 0)
              or b["opcodes"].get("R2UR", 0)):
            status = "FAIL_MASK_CHAIN_NOT_REMOVED"
        elif b["opcodes"].get("BRA.DIV", 0):
            guards = b["divergence_guards"]
            if (len(guards) == b["opcodes"]["BRA.DIV"] <= a["opcodes"].get("BRA.DIV", 0) and
                    all(guard["classification"] == "CONSTANT_FULL_MASK_SYNC_GUARD" for guard in guards)):
                status = "PARTIAL_MASK_MATERIALIZATION_REMOVED_SYNC_GUARDS_REMAIN"
            else:
                status = "FAIL_UNRESOLVED_DIVERGENCE_GUARDS"
        else:
            status = "PASS_MASK_CHAIN_REMOVED"
        routes[columns] = {"status": status, "arithmetic_opcode_counts_match": arithmetic_counts_match,
                           "arithmetic_check_scope": "main K-loop interval; excludes branch target blocks",
                           "explicit_active_mask_chain_removed": bool(a["mask_chains"]) and
                               not b["mask_chains"] and not b["opcodes"].get("VOTE.ANY", 0) and
                               not b["opcodes"].get("R2UR", 0),
                           "all_runtime_loop_divergence_guards_removed": not b["opcodes"].get("BRA.DIV", 0),
                           "off": off, "on": on}
    statuses = {item["status"] for item in routes.values()}
    if statuses == {"PASS_MASK_CHAIN_REMOVED"}:
        status, exit_code = "PASS_COMPILE_MECHANISM_GATE", 0
    elif statuses == {"NO_MECHANISM_BASELINE_ALREADY_UNIFORM"}:
        status, exit_code = "NO_MECHANISM_BASELINE_ALREADY_UNIFORM", 3
    elif statuses == {"PARTIAL_MASK_MATERIALIZATION_REMOVED_SYNC_GUARDS_REMAIN"}:
        status, exit_code = "PARTIAL_COMPILE_MECHANISM_SYNC_GUARDS_REMAIN", 1
    else:
        status, exit_code = "NOT_QUALIFIED_COMPILE_MECHANISM_GATE", 1
    result = {"status": status, "routes": routes, "objects": receipts,
              "generator_receipt_sha256": sha(root / "generated.json"),
              "common_flags": (root / "common-flags.txt").read_text(encoding="utf-8").strip(),
              "toolchain": (root / "toolchain.txt").read_text(encoding="utf-8").strip(),
              "analyzer": analyzer, "read_only": read_only,
              "numerical_validation_run": False, "performance_claim": False}
    if read_only:
        print(json.dumps(result, sort_keys=True))
    else:
        (root / "gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": status, "routes": {t: r["status"] for t, r in routes.items()},
                          "receipt": str(root / "gate.json"), "performance_claim": False}, sort_keys=True))
    return exit_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=pathlib.Path, default=pathlib.Path("/build/q6-full-warp"))
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args()
    try:
        return run(args.build_root.resolve(), args.read_only)
    except (KeyError, OSError, ValueError) as error:
        print(json.dumps({"status": "ERROR_UNQUALIFIED_GATE_INPUT", "error": str(error)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
