#ifdef FIX_CLASS
// clang-format off
FixStyle(imagecharges, FixImageCharges)
// clang-format on
#else

#ifndef LMP_FIX_IMAGE_CHARGES_H
#define LMP_FIX_IMAGE_CHARGES_H

#include "fix.h"

namespace LAMMPS_NS {

class FixImageCharges: public Fix {
 public: 
  FixImageCharges(class LAMMPS *, int, char **);
  virtual ~FixImageCharges();
  int setmask();
  virtual void init();
  void min_setup_pre_force(int);
  void setup_pre_force(int);
  void min_pre_force(int);
  void pre_force(int);
  void min_post_force(int); 
  void post_force(int);
  void post_run();

  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int,int);
  void set_arrays(int);

 protected: 
  class Region *region;
  char *pxstr, *pystr, *pzstr, *nxstr, *nystr, *nzstr;
  int *imagei;
  double *imageid;

  double pxvalue, pyvalue, pzvalue, nxvalue, nyvalue, nzvalue, scale;
  int pxvar, pyvar, pzvar, nxvar, nyvar, nzvar, scalevar;
  int pxstyle, pystyle, pzstyle, nxstyle, nystyle, nzstyle, scalestyle;
  int varflag, exclusionAtom, itype;
};
}

#endif
#endif