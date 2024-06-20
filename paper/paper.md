---
title: 'MDCraft: A python assistant for performing and analysing molecular dynamics simulations'
tags:
  - Python
  - molecular dynamics
  - trajectory analysis
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

`MDCraft` is a comprehensive Python package designed to streamline research workflows involving molecular dynamics (MD) simulations. It simplifies the entire process, from setting up and running simulations, to using sophisticated algorithm to analyze the trajectories, and post-processing tools to visualize the results, thus making computational chemistry accessible to a wider audience. At its core, `MDCraft` comprises three main components. 

First, for analysing and optimising MD simulations, the `lammps` and `openmm` modules provide simple tools so that users can explore a wide variety of large soft matter systems over a range of timescales in the LAMMPS [@thompson:2022] and OpenMM [@eastman:2017] simulation packages, respectively. These modules extends the functionality of popular MD simulation toolkits by introducing custom force fields, such as the efficient and intuitive Gaussian core model with smeared electrostatics (GCMe) [@ye:2024], incorporating problem-solving techniques such as the slab correction [@yeh:1999, @ballenegger:2009] and the method of image charges [@hautman:1989] for charged systems with a slab geometry, allowing coarse-grained MD simulations by scaling physical values by fundamental quantities (mass $m$, length $d$, energy $\epsilon$, and Boltzmann constant $k_\mathrm{B}T$), and offering feature-rich readers and writers for topologies and trajectories stored in memory-efficient formats.

Next, for analyzing the simulation trajectories, the `algorithm` and `analysis` modules offer optimized serial and multithreaded algorithms and analysis classes, respectively, for evaluating structural, thermodynamic, and dynamic properties using the generated thermodynamic state and trajectory data. The properties that can be computed include, but are not limited to, structure factors, density and potential profiles, end-to-end vector autocorrelation functions for polymers, and Onsager transport coefficients [@fong:2020] for charged species. 

Finally, the `fit` and `plot` modules facilitate post-processing of the results so that they could be suitable for scientific publications. Curve fitting and helper functions that interface with the commonly used SciPy and Matplotlib libraries are also provided.

These three aspect provide both entry-level and experienced users of MD simulations all the tools they need to perform their experiments, both simple and complex, within one succinct package.

# Statement of need
While well-established molecular dynamics analysis packages such as `MDAnalysis` [@michaud-agrawalMDAnalysisToolkitAnalysis2011] and `MDTraj` [@McGibbon2015MDTraj] have existed for a long time, their primary focus has been on the analysis of simulations, whereas `MDCraft` aims to provide the tools needed throughout the simulation process, from initialization to post-processing. In particular, the youngest of the standardized simulation packages, OpenMM, offers little in the way of an easy-to-use interface and analysis tools, with `MDCraft` filling-in these gaps. Furthermore, leveraging the modularity of OpenMM, `MDCraft` provides brand new simulation potentials (GCMe) and methods (image charges) not typically available in other simulation packages.

In addition, what distinguishes `MDCraft` with regards to its analysis toolkit is its remarkable flexibility. General users have unprecedented control over what aspects of the properties to calculate and which method to use through a plethora of well-documented built-in options in each analysis class, without having to be concerned about the underlying implementations. More advanced users, on the other hand, have the option to work directly with the algorithms for further customization. These tools have proven to be vital in a number of recent publications [@glismanMultivalentIonMediatedPolyelectrolyte2024;@manthaAdsorptionIsothermMechanism2024;@Lee2024].

# Acknowledgements
We acknowledge contributions from Alec Glisman in the development of this package. Z-G.W. acknowledges funding from Hong Kong Quantum AI Lab, AIR\@InnoHK of the Hong Kong Government.

# References