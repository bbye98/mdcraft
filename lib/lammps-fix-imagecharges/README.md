`fix imagecharges` -- LAMMPS Image Charge Fix
=============================================

This LAMMPS fix is an updated version of `fix_imagecharges.*` from
[`lammps_fixes`](https://github.com/kdwelle/lammps-fixes) and is 
compatible with the current version of LAMMPS (21 Nov 2023). It 
implements the method of image charges for a system of charged particles
confined between two parallel perfectly conducting electrodes.

Currently, only a CPU implementation with support for the OpenMP package
is available. 

Installation
============

Before compiling LAMMPS, copy the `fix_imagecharges.cpp` and
`fix_imagecharges.h` files to the `src` directory of the LAMMPS
distribution. Then, recompile LAMMPS as usual.

fix imagecharges command
========================

### Syntax

    fix ID group-ID imagecharges px py pz nx ny nz itype keyword value ...

* ID, group-ID are documented in [fix](https://docs.lammps.org/fix.html) command
* imagecharges = style name of this fix command
* px, py, pz = coordinates of a point on the image plane
* nx, ny, nz = vector normal to the image plane
* itype = atom type to be used as the image charges
* one or more keyword/value pairs may be appended

      keyword = region or scale
        region = ID of region that encompasses the real atoms
        scale value = f
          f = charge scale factor (default = 1.0)

### Examples

    fix 1 all imagecharges 0.0 0.0 0.0 0.0 0.0 1.0 4