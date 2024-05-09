`openmm-ic` — OpenMM Image Charge Plugin
========================================

This OpenMM plugin is an updated version of 
[`openmm_constV`](https://github.com/scychon/openmm_constV/)
and implements two Langevin integrators that can 
simulate slab systems of normal or Drude particles, respectively, 
confined between two parallel perfectly conducting electrodes.

Currently, only the CUDA platform is supported.

Building The Plugin
===================

This project uses [CMake](http://www.cmake.org) for its build system.
To build it, follow these steps:

1. Create a directory in which to build the plugin.
2. Set environmental variables such as `CXXFLAGS='-std=c++11'`,
   `OPENMM_CUDA_COMPILER=$(which nvcc)`, etc.
3. Run the CMake GUI or `ccmake` and specify your new directory as the
   build directory and the top level directory of this project as the 
   source directory.
4. Press "Configure".
5. Set `OPENMM_DIR` to point to the directory where OpenMM is installed 
   (usually `/usr/local/openmm` if self-compiled or `$CONDA_PREFIX` if 
   installed via conda-forge). This is needed to locate the OpenMM 
   header files and libraries.
6. Set `CMAKE_INSTALL_PREFIX` to the directory where the plugin should 
   be installed. Usually, this will be the same as `OPENMM_DIR`, so the
   plugin will be added to your OpenMM installation.
7. Make sure that `CUDA_TOOLKIT_ROOT_DIR` is set correctly and that 
   `IC_BUILD_CUDA_LIB` is enabled.
8. Press "Configure" again if necessary, then press "Generate".
9. Use the build system you selected to build and install the plugin 
   using

       make -j
       make install
       make PythonInstall

   The plugin will be installed as the `openmm-ic` Python package, which
   can be imported using `import openmm_ic`.

Python API
==========

The two integrators available are `openmm_ic.ICLangevinIntegrator` and
`openmm_ic.ICDrudeLangevinIntegrator`, and they have the same methods as
their counterparts `openmm.LangevinIntegrator` and 
`openmm.DrudeLangevinIntegrator`, respectively.

A simple example is provided below:

    from openmm import app, unit
    from openmm_ic import ICLangevinIntegrator

    # Set up or load the system and topology for the real particles
    # system = ...
    # topology = ...

    # Set up or retrieve the non-bonded force
    # nbforce = ...

    # Determine which particles belong to the wall
    # wall_indices = ...

    # Double simulation box size and mirror particle positions
    pbv = system.getDefaultPeriodicBoxVectors()
    pbv[2] *= 2
    system.setDefaultPeriodicBoxVectors(*pbv)
    dims = topology.getUnitCellDimensions()
    dims[2] *= 2
    topology.setUnitCellDimensions(dims)
    positions = np.concatenate(
        (positions, positions * np.array((1, 1, -1), dtype=int))
    )

    # Register image charges to the system, topology, and force field
    chains_ic = [topology.addChain() for _ in range(topology.getNumChains())]
    residues_ic = [topology.addResidue(f"IC_{r.name}", 
                                       chains_ic[r.chain.index])
                   for r in list(topology.residues())]
    for i, atom in enumerate(list(topology.atoms())):
        system.addParticle(0 * unit.amu)
        topology.addAtom(f"IC_{atom.name}", atom.element,
                         residues_ic[atom.residue.index])'
        q = nbforce.getParticleParameters(i)[0]
        nbforce.addParticle(
            0 if i in wall_indices else 
            -nbforce.getParticleParameters(i)[0], 
            0, 0
        )

    # Add existing particle exclusions to mirrored image charges
    for i in range(nbforce.getNumExceptions()):
        i1, i2, qq = nbforce.getExceptionParameters(i)[:3]
        nbforce.addException(N_real + i1, N_real + i2, qq, 0, 0)

    # Prevent wall particles from interacting with their mirrored image charges
    for i in wall_indices:
        nbforce.addException(i, N_real + i, 0, 0, 0)

    # Create the Langevin integrator
    temp = 300 * unit.kelvin
    fric = 1 / unit.picosecond
    dt = 1 * unit.femtosecond
    integrator = ICLangevinIntegrator(temp, fric, dt)

Alternatively, if you have [MDHelper](https://github.com/bbye98/mdhelper)
installed, you can use the `mdhelper.openmm.system.image_charges()` 
function to achieve the same result as the code snippet above.

Citing This Work
================
Any work that uses this plugin should cite the following publication:

C. Y. Son and Z.-G. Wang, Image-Charge Effects on Ion Adsorption near 
Aqueous Interfaces, Proc. Natl. Acad. Sci. U.S.A. 118, e2020615118 
(2021). https://doi.org/10.1073/pnas.2020615118