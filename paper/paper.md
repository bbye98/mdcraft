---
title: 'MDCraft: A Python assistant for performing and analyzing molecular dynamics simulations of soft matter systems'
tags:
  - Python
  - molecular dynamics
  - trajectory analysis
  - soft matter
authors:
  - name: Benjamin B. Ye
    orcid: 0000-0003-0253-6311
    corresponding: true
    affiliation: 1
  - name: Pierre J. Walker
    orcid: 0000-0001-8628-6561
    affiliation: "1, 2"
  - name: Zhen-Gang Wang
    orcid: 0000-0002-3361-6114
    affiliation: 1
affiliations:
 - name: Division of Chemistry and Chemical Engineering, California Institute of Technology, Pasadena, California 91125, United States
   index: 1
 - name: Department of Chemical Engineering, Imperial College, London SW7 2AZ, United Kingdom
   index: 2
date: June __, 2024
bibliography: paper.bib
---

# Summary

MDCraft is a comprehensive Python package designed to enhance research workflows involving molecular dynamics (MD) simulations. It streamlines the entire process—from setting up and executing simulations to analyzing trajectories using sophisticated algorithms and visualizing results—making computational chemistry more accessible to a broader audience. At its core, MDCraft comprises three principal components.

First, the `openmm` module provides user-friendly tools to initialize, optimize, and run simulations, enabling the exploration of various large soft matter systems across different timescales. This module extend the functionality of the OpenMM [@eastman_openmm_2017] simulation package by introducing custom force fields, such as the efficient and intuitive Gaussian core model with smeared electrostatics (GCMe) [@ye_gcme_2024]; incorporating advanced techniques like the slab correction [@yeh_ewald_1999;@ballenegger_simulations_2009] and the method of image charges [@hautman_molecular_1989] for charged systems with slab geometries; facilitating coarse-grained MD simulations by scaling physical values by the fundamental quantities (mass $m$, length $d$, energy $\epsilon$, and Boltzmann constant $k_\mathrm{B}T$); and offering feature-rich readers and writers for topologies and trajectories stored in memory-efficient formats (such as NetCDF).

Second, the `algorithm` and `analysis` modules offer optimized serial and multithreaded algorithms and analysis classes for evaluating structural, thermodynamic, and dynamic properties using thermodynamic state and trajectory data. The analysis classes provide properties including, but not limited to, static and dynamic structure factors [@faberTheoryElectricalProperties1965;@ashcroftStructureBinaryLiquid1967;@rogMoldynProgramPackage2003], density and potential profiles, end-to-end vector autocorrelation functions for polymers, and Onsager transport coefficients [@rubinsteinPolymerPhysics2003;@fong_onsager_2020]. The algorithms provide the underlying tools used to perform analysis and are intended to be easily extensible by more-advanced users. These modules are not limited to OpenMM only and can be used with simulation run in other packages such as LAMMPS [@thompson_lammps_2022] and GROMACS [@abrahamGROMACSHighPerformance2015].

Finally, the `fit` and `plot` modules simplify the post-processing and visualization of data, aiding in the creation of aesthetically pleasing figures for scientific publications. These modules consist of models for curve fitting and helper functions that interface seamlessly with the commonly used SciPy [@virtanen_scipy_2020] and Matplotlib [@hunter_matplotlib_2007] libraries.

Together, these modules provide both novice and experienced MD simulation users with a comprehensive set of tools necessary to conduct computer experiments ranging from simple to complex, all within a single, succinct package.

# Statement of need

Although established MD analysis packages such as MDAnalysis [@michaudagrawal_mdanalysis_2011] and MDTraj [@mcgibbon_mdtraj_2015] have been around for a considerable time, they primarily focus on the post-simulation analysis. In contrast, MDCraft is designed to provide comprehensive support throughout the entire simulation process, from initialization to post-processing. 

MDCraft is tightly integrated with OpenMM, a relatively new simulation toolkit that has seen a surge in popularity in recent years due to its class-leading performance and flexibility through support for custom intermolecular forces and integrators for equations of motion. Due to its age and design philosophy, OpenMM offers comparatively fewer choices of pair potentials and external forces, and no built-in analysis support. MDCraft fills this gap in two ways. First, the `openmm` module leverages the modularity of OpenMM to provide a suite of custom force fields, problem-solving tools, trajectory readers and writers, and utility functions for unit reduction, topology transformations, and performance optimizations that are not typically available in other simulation packages. Of special significance is the support for the GCMe force field which, as demonstrated in a recent article[@ye_gcme_2024], provides substantial acceleration compared to other force fields while also remaining physically meaningful. Then, the classes in the `analysis` module enable computing common structural, thermodynamic, and dynamic properties using the topology, trajectory, and state data generated by OpenMM (or other simulation packages).

The `analysis` module also stands out due to the remarkable flexibility it affords its end users. General users have unprecedented control over what aspects of the properties to calculate and which method to employ through a plethora of well-documented built-in options in each analysis class, without having to be concerned about the underlying implementations. More advanced users, on the other hand, have the option to work directly with the algorithms in the `algorithms` module for further customization. These analysis functions and classes have proven indispensable in several recent publications [@glisman_multivalent_2024;@mantha_adsorption_2024;@lee_molecular_2024].

The application of MDCraft extends across various domains within computational chemistry and materials science. Researchers can utilize it to study the low-level mechanisms involved in supercapacitors, polymer gels, drug delivery systems, and nanomaterial synthesis, thus highlighting its versatility and broad applicability in cutting-edge scientific research.

# Acknowledgements

We acknowledge contributions from Alec Glisman and Dorian Bruch in the development of this package and financial support from Hong Kong Quantum AI Lab, AIR\@InnoHK of the Hong Kong Government.

# References