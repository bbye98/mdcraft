"""
OpenMM utility functions
========================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains OpenMM-related utility functions.
"""

from datetime import datetime
import itertools
import logging
from typing import Union

import numpy as np
import openmm
from openmm import unit

def _create_context(
        system: openmm.System, integrator: openmm.Integrator,
        positions: np.ndarray[float], platform: openmm.Platform,
        properties: dict) -> openmm.Context:

    r"""
    Creates an OpenMM Context by cloning the Integrator passed to this
    function. Useful for benchmarking different simulation systems.

    Parameters
    ----------
    system : `openmm.System`
        OpenMM molecular system.

    integrator : `openmm.Integrator`
        OpenMM integrator or thermostat.

    positions : `np.ndarray` or `unit.Quantity`
        Initial positions of the :math:`N` particles in the system.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    platform : `openmm.Platform`
        OpenMM platform.

    properties : `dict`
        Dictionary of platform-specific properties.

    Returns
    -------
    context : `openmm.Context`
        OpenMM simulation context.
    """

    integrator = openmm.XmlSerializer.clone(integrator)
    context = openmm.Context(system, integrator, platform, properties)
    context.setPositions(positions)
    return context

def _benchmark_integrator(context: openmm.Context, steps: int) -> float:

    """
    Benchmarks the performance of an OpenMM Integrator.

    Parameters
    ----------
    context : `openmm.Context`
        OpenMM simulation context.

    Returns
    -------
    time : `float`
        Elapsed time, in seconds.
    """

    start = datetime.now()
    context.getIntegrator().step(steps)
    return (datetime.now() - start).total_seconds()

def optimize_pme(
        system: openmm.System, integrator: openmm.Integrator,
        positions: Union[np.ndarray[float], unit.Quantity],
        platform: openmm.Platform, properties: dict,
        min_cutoff: Union[float, unit.Quantity],
        max_cutoff: Union[float, unit.Quantity], *,
        pmeforce: openmm.NonbondedForce = None, cpu_pme: bool = True,
        target: float = 10, target_std: float = None, window: int = 3,
        fastest: int = 5, rerun: int = 2, verbose: bool = True
    ) -> tuple[unit.Quantity, bool]:

    """
    Runs a series of simulations using different parameters to determine
    the optimal configuration for evaluating electrostatic interactions
    with the particle mesh Ewald (PME) method on a GPU (CUDA or OpenCL).

    The cutoff distance for the Coulomb potential can be freely varied
    with no accuracy penalty since OpenMM automatically selects internal
    parameters to satisfy the specified error tolerance. However, it may
    affect the accuracy of other nonbonded interactions, so care must be
    taken to ensure the optimal Coulomb potential cutoff is compatible
    with the other pair potentials in the system.

    In certain cases, OpenMM can perform better with the reciprocal
    space calculations being done on the CPU while the direct space
    calculations are being evaluated on the GPU.

    Parameters
    ----------
    system : `openmm.System`
        OpenMM molecular system.

    integrator : `openmm.Integrator`
        OpenMM integrator or thermostat.

    positions : `np.ndarray` or `unit.Quantity`
        Initial positions of the :math:`N` particles in the system.

        **Shape**: :math:`(N,\\,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    platform : `openmm.Platform`
        OpenMM platform.

    properties : `dict`
        Dictionary of platform-specific properties.

    min_cutoff : `float` or `unit.Quantity`
        Minimum cutoff distance to test.

        **Reference unit**: :math:`\\mathrm{nm}`.

    max_cutoff : `float` or `unit.Quantity`
        Maximum cutoff distance to test.

        **Reference unit**: :math:`\\mathrm{nm}`.

    pmeforce : `openmm.NonbondedForce` or `openmm.AmoebaMultipoleForce`, \
    keyword-only, optional
        Pair potential with electrostatic interactions that are
        evaluated using PME.

    cpu_pme : `bool`, keyword-only, default: :code:`True`
        Determines whether CPU PME should be benchmarked.

    target : `float`, keyword-only, default: :code:`10`
        Target simulation time for each test run, in seconds.

    target_std : `float`, keyword-only, optional
        Allowed variability for `target`, in seconds. If set to a value
        that is too small, the target simulation time may never be
        satisfied. If not specified, it is set to 10% of `target`.

    window : `int`, keyword-only, default: :code:`3`
        Number of previous runs to look at before deciding whether to
        stop testing larger cutoffs.

    fastest : `int`, keyword-only, default: :code:`5`
        Number of fastest configurations to retest before deciding on
        the optimal cutoff distance and hardware (architecture).

    rerun : `int`, keyword-only, default :code:`2`
        Number of reruns to complete for the fastest preliminary runs.

    verbose : `bool`, keyword-only, default: :code:`True`
        Determines whether detailed progress is shown.

    Returns
    -------
    cutoff : `unit.Quantity`
        Optimal cutoff distance.

        **Reference unit**: :math:`\\mathrm{nm}`.

    cpu_pme : `bool`
        Specifies whether to use the CPU to perform reciprocal space
        calculations.
    """

    # Set up logger
    logging.basicConfig(format="{asctime} | {levelname:^8s} | {message}",
                        style="{",
                        level=logging.INFO if verbose else logging.WARNING)

    # Get information about the pair potential of interest
    if pmeforce is None:
        for force in system.getForces():
            if isinstance(force, (openmm.NonbondedForce,
                                  openmm.AmoebaMultipoleForce)):
                pmeforce = force
                break
    if pmeforce.getNonbondedMethod() != openmm.NonbondedForce.PME:
        raise ValueError("The provided (or guessed) pair potential is "
                        "not being evaluated using the particle mesh "
                        "Ewald (PME) method.")
    cpu_pme &= isinstance(pmeforce, openmm.NonbondedForce) \
               and platform.supportsKernels(["CalcPmeReciprocalForce"])
    tol = pmeforce.getEwaldErrorTolerance()

    # Determine the number of timesteps to run for each cutoff distance
    logging.info("Determining a reasonable number of timesteps for PME optimizer...")
    pmeforce.setCutoffDistance(np.sqrt(min_cutoff * max_cutoff))
    properties["UseCpuPme"] = "false"
    context = _create_context(system, integrator, positions, platform,
                              properties)
    if target_std is None:
        target_std = 0.1 * target
    time_width = max(9, np.ceil(np.log10(target)).astype(int) + 7)
    lb, ub = target - target_std, target + target_std
    steps, time = 20, 0
    while True:
        time = _benchmark_integrator(context, steps)
        logging.info(f"  GPU: {steps:14,} ts "
                     f"===> {time:{time_width}.5f} s elapsed")
        if lb < time < ub:
            break
        steps = int(target * steps / time)
    if cpu_pme:
        properties["UseCpuPme"] = "true"
        context = _create_context(system, integrator, positions, platform,
                                  properties)
        steps_cpu, time = 20, 0
        while True:
            time = _benchmark_integrator(context, steps_cpu)
            logging.info(f"  CPU: {steps_cpu:14,} ts "
                         f"===> {time:{time_width}.5f} s elapsed")
            if lb < time < ub:
                break
            steps_cpu = int(target * steps_cpu / time)
        steps = min(steps, steps_cpu)
    steps = np.round(steps, 2 - np.ceil(np.log10(steps)).astype(int))
    logging.info(f"Starting PME optimizer (using {steps:,} timesteps)...")

    # Build list of cutoff distances to test
    if isinstance(min_cutoff, unit.Quantity):
        min_cutoff = min_cutoff.value_in_unit(unit.nanometer)
    if isinstance(max_cutoff, unit.Quantity):
        max_cutoff = max_cutoff.value_in_unit(unit.nanometer)
    cutoffs = {"gpu": {min_cutoff}}
    if cpu_pme:
        cutoffs["cpu"] = {min_cutoff}

    for dim in [v[i].value_in_unit(unit.nanometer)
                for i, v in enumerate(system.getDefaultPeriodicBoxVectors())]:

        # Iterate through possible grid sizes
        for n_mesh in itertools.count(start=5):

            # Check if grid size is legal for FFT
            check = n_mesh
            for factor in (2, 3, 5, 7):
                while check > 1 and check % factor == 0:
                    check /= factor
            if check not in (1, 11, 13):
                continue

            # Compute smallest cutoff that will give the current grid
            # size
            alpha = 1.5 * n_mesh * tol ** 0.2 / dim
            cutoff = np.round(np.sqrt(-np.log(2 * tol) / alpha), 3)
            if cutoff < min_cutoff:
                break
            if cutoff < max_cutoff:
                if cpu_pme:
                    cutoffs["cpu"].add(cutoff)
                if check == 1:
                    cutoffs["gpu"].add(cutoff)

    # Get preliminary times for the different architectures and cutoffs
    cutoff_width = max(7, np.ceil(
        np.log10(max(max(v) for v in cutoffs.values()))
    ).astype(int) + 6)
    times = {}
    for arch, cut in cutoffs.items():
        cutoffs[arch] = np.array(sorted(cut))
        times[arch] = np.empty(cutoffs[arch].shape)
        times[arch][:] = np.nan
        for i, cutoff in enumerate(cutoffs[arch]):
            pmeforce.setCutoffDistance(cutoff)
            properties["UseCpuPme"] = str(arch == "cpu").lower()
            context = _create_context(system, integrator, positions, platform,
                                      properties)
            times[arch][i] = _benchmark_integrator(context, steps)
            logging.info(f"  {arch.upper()}: {cutoff:{cutoff_width}.4f} nm cutoff "
                         f"===> {times[arch][i]:{time_width}.5f} s elapsed")

            # Stop iteration if simulation is continuously getting slower
            if i > window and np.all(times[arch][i - window:i] >
                                     times[arch][i - window - 1:i - 1]):
                break
    best = sorted([t, c, a] for a in times.keys()
                            for c, t in zip(cutoffs[a], times[a]))[:fastest]

    # Rerun the fastest configurations to ensure correct results
    for i, (time, cutoff, arch) in enumerate(best):
        pmeforce.setCutoffDistance(cutoff)
        properties["UseCpuPme"] = str(arch == "cpu").lower()
        context = _create_context(system, integrator, positions, platform,
                                  properties)

        # Replace preliminary time with median time from reruns
        best[i][0] = sorted(
            (time, *[_benchmark_integrator(context, steps) for _ in range(rerun)])
        )[1]
    best.sort()

    # Display fastest configurations and timings
    time_width = 8 + 2 * np.ceil(max(0, time_width - 8) // 2).astype(int)
    cutoff_width = 11 + 2 * np.ceil(max(0, cutoff_width - 11) // 2).astype(int)
    logging.info(f"PME optimization completed.\n"
                 f"   Rank | {'Time (s)':^{time_width}} "
                 f"| {'Cutoff (nm)':^{cutoff_width}} | CPU PME\n"
                 f"  ------|{'-' * (time_width + 2)}"
                 f"|{'-' * (cutoff_width + 2)}|---------\n  " +
                 "\n  ".join(f" {i + 1:>4} | {time:{time_width}.5f} |"
                             f" {cutoff:{cutoff_width}.4f} | {arch == 'cpu'}"
                             for i, (time, cutoff, arch) in enumerate(best)))

    # Return optimized configuration
    return best[0][1] * unit.nanometer, arch == "cpu"