#!/usr/bin/env python3

import copy
import pickle
from pathlib import Path


AMPLITUDES_URAD = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    15, 20, 25, 30, 35, 40, 45, 50,
]

BASE_INPUT = "scan_input.pkl"


def main():

    with open(BASE_INPUT, "rb") as f:
        base = pickle.load(f)

    for amplitude_urad in AMPLITUDES_URAD:

        directory = Path(
            f"amp_{amplitude_urad:02d}urad"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        (directory / "results").mkdir(
            exist_ok=True,
        )

        (directory / "logs").mkdir(
            exist_ok=True,
        )

        config = copy.deepcopy(base)

        config["shaker_amplitude"] = (
            amplitude_urad * 1e-6
        )

        config["shaker_amplitude_urad"] = (
            amplitude_urad
        )

        with open(
            directory / "scan_input.pkl",
            "wb",
        ) as f:
            pickle.dump(
                config,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(
            f"Prepared {amplitude_urad:2d} urad"
        )


if __name__ == "__main__":
    main()
