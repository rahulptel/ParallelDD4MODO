#!/usr/bin/env python3
from pathlib import Path
import os
import re

BACKEND = "gpu"
INCLUDE_CPU_KERNELS = 0
CPU_WORKERS = 1
FORCED_METHOD = 1

KNAPSACK_MIN_VARS = 40
SETPACKING_MIN_VARS = 150
TSP_MIN_CITIES = 15


def iter_dat_files(directory: Path):
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.dat") if p.is_file())


def parse_knapsack_nvars(name: str):
    # Patterns: KP_p-4_n-40_ins-1.dat OR knapsack-100-4-1000-116.dat
    m = re.match(r"^KP_p-\d+_n-(\d+)_ins-\d+\.dat$", name)
    if m:
        return int(m.group(1))
    m = re.match(r"^knapsack-(\d+)-\d+-\d+-\d+\.dat$", name)
    if m:
        return int(m.group(1))
    return None


def parse_setpacking_nvars(name: str):
    # Pattern: bp-150-30-3-10-13932.dat
    m = re.match(r"^bp-(\d+)-\d+-\d+-\d+-\d+\.dat$", name)
    return int(m.group(1)) if m else None


def parse_tsp_cities(name: str):
    # Pattern: tsp-nobj3-ncities15-seed108.dat
    m = re.match(r"^tsp-nobj\d+-ncities(\d+)-seed\d+\.dat$", name)
    return int(m.group(1)) if m else None


def append_case(lines, binary, instance, problem_type, method, dominance):
    if BACKEND == "gpu":
        lines.append(
            f"{binary} {instance} {problem_type} {method} {dominance} --backend gpu --save-frontier --save-stats"
        )
        return

    if INCLUDE_CPU_KERNELS:
        for cpu_kernel in (1, 3):
            lines.append(
                f"{binary} {instance} {problem_type} {method} {dominance} --backend cpu --cpu-threads {CPU_WORKERS} --cpu-kernel {cpu_kernel} --save-frontier --save-stats"
            )
    else:
        lines.append(
            f"{binary} {instance} {problem_type} {method} {dominance} --backend cpu --save-frontier --save-stats"
        )


def main():
    script_dir = Path(__file__).resolve().parent
    farm_name = script_dir.name

    project_root = Path(os.environ.get("PROJECT_ROOT", "/home/rahulpat/scratch/cuMODD")).resolve()
    table_path = Path(os.environ.get("TABLE_PATH", str(script_dir / "table.dat"))).resolve()
    nobjs_min = int(os.environ.get("NOBJS_MIN", "3"))
    nobjs_max = int(os.environ.get("NOBJS_MAX", "7"))

    local_data_default = script_dir.parent.parent / "data"
    if "SOURCE_DATA_ROOT" in os.environ:
        source_data_base = Path(os.environ["SOURCE_DATA_ROOT"]).resolve()
    elif local_data_default.is_dir():
        source_data_base = local_data_default.resolve()
    else:
        source_data_base = (project_root / "data").resolve()

    binary_base = project_root / "resources" / "bin" / farm_name
    target_data_base = project_root / "data"

    lines = []
    for nobjs in range(nobjs_min, nobjs_max + 1):
        source_data_root = source_data_base / str(nobjs)
        if not source_data_root.is_dir():
            continue

        binary = binary_base / f"multiobj_nobjs{nobjs}"

        if FORCED_METHOD == 0:
            method_knapsack = 3
            method_binproblem = 3
            method_tsp = 3
        else:
            method_knapsack = FORCED_METHOD
            method_binproblem = FORCED_METHOD
            method_tsp = FORCED_METHOD

        for path in iter_dat_files(source_data_root / "knapsack"):
            nvars = parse_knapsack_nvars(path.name)
            if nvars is None or nvars < KNAPSACK_MIN_VARS:
                continue
            target_instance = target_data_base / str(nobjs) / "knapsack" / path.name
            append_case(lines, binary, target_instance, 1, method_knapsack, 1)

        for path in iter_dat_files(source_data_root / "binproblem"):
            nvars = parse_setpacking_nvars(path.name)
            if nvars is None or nvars < SETPACKING_MIN_VARS:
                continue
            target_instance = target_data_base / str(nobjs) / "binproblem" / path.name
            append_case(lines, binary, target_instance, 2, method_binproblem, 0)

        for path in iter_dat_files(source_data_root / "tsp"):
            ncities = parse_tsp_cities(path.name)
            if ncities is None or ncities < TSP_MIN_CITIES:
                continue
            target_instance = target_data_base / str(nobjs) / "tsp" / path.name
            append_case(lines, binary, target_instance, 3, method_tsp, 0)

    table_path.write_text("\n".join(str(line) for line in lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Generated {table_path}")
    print(f"{len(lines)} {table_path}")


if __name__ == "__main__":
    main()
