#!/usr/bin/env python3

"""
Run one FESTA2 simulation at a single shaker frequency.
"""

import argparse
import pickle
import time
from pathlib import Path

import at

import festa2


def main():
    parser = argparse.ArgumentParser(
        description="Run one FESTA2 frequency simulation."
    )

    parser.add_argument(
        "index",
        type=int,
        help="Index of the frequency in the input file.",
    )

    parser.add_argument(
        "scan_dir",
        type=str,
        help="Directory containing scan_input.pkl.",
    )

    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)

    input_file = scan_dir / "scan_input.pkl"
    output_dir = scan_dir / "results"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with input_file.open("rb") as file:
        config = pickle.load(file)

    tunes = config["tunes"]

    if args.index < 0 or args.index >= len(tunes):
        raise IndexError(
            f"Frequency index {args.index} is outside "
            f"the valid range 0-{len(tunes) - 1}."
        )

    tune = float(
        tunes[args.index]
    )

    ring_at = at.load_lattice(
        config["lattice_file"],
        key=config["lattice_key"],
    )

    _, ring_at_rot = festa2.xs_from_at(
        ring_at
    )

    _, frr = at.fast_ring(
        ring_at_rot
    )

    shaker = festa2.Shaker(
        tune=tune,
        amplitude=config["shaker_amplitude"],
        phase=config["shaker_phase"],
    )

    start = time.perf_counter()

    result = festa2.simulate_one_frequency(
        tune=tune,
        frr=frr,
        part_initial=config["part_initial"],
        shaker_initial=shaker,
        OAM=config["OAM"],
        R0=config["R0"],
        closed_orbit=config["closed_orbit"],
        nturns=config["nturns"],
        save_every=config["save_every"],
        average_blocks=config["average_blocks"],
    )

    elapsed = time.perf_counter() - start

    result.pop(
        "part_final",
        None,
    )

    result["index"] = args.index

    result["tune_offset"] = float(
        config["tune_offsets"][args.index]
    )

    if "spin_tunes_unwrapped" in config:
        result["spin_tune"] = float(
            config["spin_tunes_unwrapped"][args.index]
        )

    if "energies_GeV" in config:
        result["energy_GeV"] = float(
            config["energies_GeV"][args.index]
        )

    if "shaker_frequency_kHz" in config:
        result["shaker_frequency_kHz"] = float(
            config["shaker_frequency_kHz"][args.index]
        )

    result["elapsed_time"] = elapsed

    result["shaker_amplitude"] = float(
        config["shaker_amplitude"]
    )

    if "shaker_amplitude_urad" in config:
        result["shaker_amplitude_urad"] = float(
            config["shaker_amplitude_urad"]
        )

    output_file = (
        output_dir
        / f"frequency_{args.index:05d}.pkl"
    )

    temporary_file = output_file.with_suffix(
        ".tmp"
    )

    with temporary_file.open("wb") as file:
        pickle.dump(
            result,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    temporary_file.replace(
        output_file
    )

    print(
        f"Completed scan_dir={scan_dir}, "
        f"index={args.index}, "
        f"tune={tune:.10f}, "
        f"elapsed={elapsed:.3f} s"
    )

    print(
        f"Result written to {output_file}"
    )


if __name__ == "__main__":
    main()
