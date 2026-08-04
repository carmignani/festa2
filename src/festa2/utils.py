"""
General utility functions for FESTA2.
"""

import time


def tic():
    """
    Start a high-resolution timer.

    Returns
    -------
    float
        Current value of the performance counter.
    """
    return time.perf_counter()


def toc(t0):
    """
    Stop a timer started with :func:`tic`.

    Parameters
    ----------
    t0 : float
        Initial time returned by ``tic()``.

    Returns
    -------
    float
        Elapsed time in seconds.
    """
    dt = time.perf_counter() - t0
    print(f"Elapsed time: {dt:.3f} s")
    return dt
