---
title: 'MDCraft: A Python assistant for performing and analysing molecular dynamics simulations of soft matter systems'
tags:
  - Python
  - Molecular Dynamics
  - Trajectory Analysis
  - Soft Matter
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
  
date:
bibliography: paper.bib
---

# Summary

`MDCraft` is a comprehensive Python package designed to enhance research workflows involving molecular dynamics (MD) simulations. It streamlines the entire process, from the setup and execution of simulations, to the analysis of trajectories using sophisticated algorithms, and post-processing tools for visualizing results, thereby making computational chemistry more accessible to a broader audience. At its core, MDCraft comprises three principal components.

First, for the analysis and optimization of MD simulations, the `lammps` and `openmm` modules provide user-friendly tools enabling the exploration of a wide variety of large soft matter systems over a range of timescales in the LAMMPS [@thompson:2022] and OpenMM [@eastman:2017] simulation packages, respectively. These modules extend the functionality of popular MD simulation toolkits by introducing custom force fields, such as the efficient and intuitive Gaussian core model with smeared electrostatics (GCMe) [@ye:2024], incorporating advanced techniques like slab correction [@yeh:1999;@ballenegger:2009] and the method of image charges [@hautman:1989] for charged systems with a slab geometry, facilitating coarse-grained MD simulations by scaling physical values by fundamental quantities (mass $m$, length $d$, energy $\epsilon$, and Boltzmann constant $k_\mathrm{B}T$), and offering feature-rich readers and writers for topologies and trajectories stored in memory-efficient formats.

Second, for the analysis of simulation trajectories, the `algorithm` and `analysis` modules offer optimized serial and multithreaded algorithms and analysis classes, respectively, for evaluating structural, thermodynamic, and dynamic properties using the generated thermodynamic state and trajectory data. The properties that can be computed include, but are not limited to, structure factors, density and potential profiles, end-to-end vector autocorrelation functions for polymers, and Onsager transport coefficients [@fong:2020]. 

Finally, the `fit` and `plot` modules facilitate the post-processing of results ensuring they are suitable for scientific publications. These modules include curve fitting and helper functions that interface seamlessly with commonly used libraries such as SciPy [@2020SciPy-NMeth] and Matplotlib [@Hunter:2007] libraries are also provided.

These three components collectively provide both novice and experienced MD simulation users with a comprehensive set of tools required to conduct experiments, ranging from simple to complex, within a single succinct package.

# Statement of need
While well-established molecular dynamics analysis packages such as `MDAnalysis` [@michaud-agrawalMDAnalysisToolkitAnalysis2011] and `MDTraj` [@McGibbon2015MDTraj] have existed for some time, their primary focus has been on the post-simulation analysis. In contrast, MDCraft aims to provide tools for the entire simulation process, from initialization to post-processing. In particular, the youngest of the standardized simulation packages, OpenMM, offers little in the way of an easy-to-use interface and analysis tools, with `MDCraft` filling-in these gaps. Notably, the relatively new OpenMM simulation package lacks an intuitive interface and comprehensive analysis tools, a gap that MDCraft aims to fill. By leveraging the modularity of OpenMM, MDCraft introduces novel simulation potentials (GCMe) and methods (image charges) not typically available in other simulation packages.

Additionally, MDCraft stands out due to its remarkable flexibility within its analysis toolkit. General users have unprecedented control over what aspects of the properties to calculate and which method to employed, through a plethora of well-documented built-in options in each analysis class, without having to be concerned about the underlying implementations. Conversely, advanced users have the option to work directly with the algorithms for further customization. These tools have proven indispensable in several recent publications [@glismanMultivalentIonMediatedPolyelectrolyte2024;@manthaAdsorptionIsothermMechanism2024;@Lee2024].

The application of MDCraft extends across various domains within computational chemistry and materials science. Researchers can utilize it to study the low-level mechanisms involved in supercapacitors, polymer gels and XX, thus highlighting its versatility and broad applicability in cutting-edge scientific research.

# Acknowledgements
We acknowledge contributions from Alec Glisman in the development of this package. Z-G.W. acknowledges funding from Hong Kong Quantum AI Lab, AIR\@InnoHK of the Hong Kong Government.

# References