#!/usr/bin/env python3

"""
Merge all FESTA2 amplitude-scan results.

For each amp_XXurad directory, this script reads:
    scan_input.pkl
    results/frequency_XXXXX.pkl

and writes:
    scan_merged.pkl
"""

import pickle
from pathlib import Path


def merge_one_scan(scan_dir: Path):

    input_file = scan_dir / "scan_input.pkl"
    results_dir = scan_dir / "results"
    output_file = scan_dir / "scan_merged.pkl"

    with input_file.open("rb") as f:
        scan_input = pickle.load(f)

    tunes = scan_input["tunes"]
    n_expected = len(tunes)

    scan_results = []
    missing_indices = []

    for index in range(n_expected):

        result_file = (
            results_dir
            / f"frequency_{index:05d}.pkl"
        )

        if not result_file.exists():
            missing_indices.append(index)
            continue

        with result_file.open("rb") as f:
            result = pickle.load(f)

        if result.get("index") != index:
            raise ValueError(
                f"{scan_dir}: wrong index in {result_file}. "
                f"Expected {index}, got {result.get('index')}"
            )

        scan_results.append(result)

    scan_results.sort(
        key=lambda r: r["index"]
    )

    merged = {
        "scan_results": scan_results,
        "scan_input": scan_input,
        "tunes": scan_input["tunes"],
        "tune_offsets": scan_input["tune_offsets"],
        "missing_indices": missing_indices,
        "n_expected": n_expected,
        "n_loaded": len(scan_results),
    }

    temporary_file = output_file.with_suffix(".tmp")

    with temporary_file.open("wb") as f:
        pickle.dump(
            merged,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    temporary_file.replace(output_file)

    return {
        "scan_dir": scan_dir.name,
        "n_expected": n_expected,
        "n_loaded": len(scan_results),
        "n_missing": len(missing_indices),
        "output_file": output_file,
    }


def main():

    scan_dirs = sorted(
        Path(".").glob("amp_*urad")
    )

    if not scan_dirs:
        raise RuntimeError(
            "No amp_*urad directories found."
        )

    print()
    print("Merging amplitude scans")
    print("-----------------------")

    total_missing = 0

    for scan_dir in scan_dirs:

        info = merge_one_scan(scan_dir)

        total_missing += info["n_missing"]

        print(
            f"{info['scan_dir']:12s}  "
            f"{info['n_loaded']:4d}/"
            f"{info['n_expected']:4d}  "
            f"missing={info['n_missing']:3d}"
        )

    print()
    print(f"Scans merged : {len(scan_dirs)}")
    print(f"Total missing: {total_missing}")

    if total_missing == 0:
        print("All scans merged successfully.")


if __name__ == "__main__":
    main()
