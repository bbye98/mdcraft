#include <string.h>
#include <stdlib.h>
#include "fix_imagecharges.h"
#include "atom.h"
#include "atom_vec.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "memory.h"
#include "region.h"
#include "variable.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum {
  NONE,
  CONSTANT,
  EQUAL,
  ATOM
};

FixImageCharges::FixImageCharges(LAMMPS *lmp, int narg, char **arg):
  Fix(lmp, narg, arg),
  pxstr(nullptr), pystr(nullptr), pzstr(nullptr),
  nxstr(nullptr), nystr(nullptr), nzstr(nullptr),
  region(nullptr), idregion(nullptr), scalestr(nullptr), imagei(nullptr), imageid(nullptr) {
    if (narg < 10) error->all(FLERR, "Illegal fix imagecharges command: not enough arguments");

    // initialize the array to keep track of image charge associations
    memory->create(imagei, atom->nmax + 2, "imagecharges::imagei");
    memory->create(imageid, atom->nmax + 2, "imagecharges::imageid");

    // first three arguments define a point on the plane
    if (strstr(arg[3], "v_") == arg[3]) pxstr = utils::strdup(&arg[3][2]);
    else {
      pxvalue = utils::numeric(FLERR, arg[3], false, lmp);
      pxstyle = CONSTANT;
    }

    if (strstr(arg[4], "v_") == arg[4]) pystr = utils::strdup(&arg[4][2]);
    else {
      pyvalue = utils::numeric(FLERR, arg[4], false, lmp);
      pystyle = CONSTANT;
    }

    if (strstr(arg[5], "v_") == arg[5]) pzstr = utils::strdup(&arg[5][2]);
    else {
      pzvalue = utils::numeric(FLERR, arg[5], false, lmp);
      pzstyle = CONSTANT;
    }

    // next three arguments define a vector normal to the plane
    if (strstr(arg[6], "v_") == arg[6]) nxstr = utils::strdup(&arg[6][2]);
    else {
      nxvalue = utils::numeric(FLERR, arg[6], false, lmp);
      nxstyle = CONSTANT;
    }

    if (strstr(arg[7], "v_") == arg[7]) nystr = utils::strdup(&arg[7][2]);
    else {
      nyvalue = utils::numeric(FLERR, arg[7], false, lmp);
      nystyle = CONSTANT;
    }

    if (strstr(arg[8], "v_") == arg[8]) nzstr = utils::strdup(&arg[8][2]);
    else {
      nzvalue = utils::numeric(FLERR, arg[8], false, lmp);
      nzstyle = CONSTANT;
    }

    // itype -- index for the image charge types
    if (strstr(arg[9], "v_") == arg[9]) {
      int itypevar = input->variable->find(&arg[8][2]);
      if (itypevar < 0) error->all(FLERR, "Variable itype for fix imagecharges does not exist");
      if (input->variable->equalstyle(itypevar)) itype = input->variable->compute_equal(itypevar);
      else error->all(FLERR, "Variable itype for fix imagecharges has invalid style");
    } else itype = utils::inumeric(FLERR, arg[9], false, lmp);

    // optional arguments
    scale = 1.0;
    int iarg = 10;
    while (iarg < narg) {
      if (strcmp(arg[iarg], "region") == 0) {
        if (iarg + 2 > narg) error->all(FLERR, "Illegal fix imagecharges command");
        region = domain->get_region_by_id(arg[iarg + 1]);
        if (!region) error->all(FLERR, "Region {} for fix imagecharges does not exist", arg[iarg + 1]);
        idregion = utils::strdup(arg[iarg + 1]);
        iarg += 2;
      } else if (strcmp(arg[iarg], "scale") == 0) {
        if (iarg + 2 > narg) error->all(FLERR, "Illegal fix imagecharges command");
        if (strstr(arg[iarg + 1], "v_") == arg[iarg + 1]) scalestr = utils::strdup(&arg[iarg + 1][2]);
        else {
            scale = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            scalestyle = CONSTANT;
        }
        iarg += 2;
      } else error->all(FLERR, "Illegal fix imagecharges command");
    }

    // this fix produces per-atom information
    peratom_flag = 1;
    peratom_freq = 1;
    vector_atom = imageid;
    atom->add_callback(0);
  }

FixImageCharges::~FixImageCharges() {
  // destructor -- free up any arrays of pointers
  delete[] pxstr;
  delete[] pystr;
  delete[] pzstr;
  delete[] nxstr;
  delete[] nystr;
  delete[] nzstr;
  delete[] scalestr;
  delete[] idregion;
  memory->destroy(imagei);
  memory->destroy(imageid);
  atom->delete_callback(id, 0);
}

int FixImageCharges::setmask() {
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= MIN_PRE_FORCE;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  mask |= POST_RUN;
  return mask;
}

void FixImageCharges::init() {
  // check variables
  if (pxstr) {
    pxvar = input->variable->find(pxstr);
    if (pxvar < 0) error->all(FLERR, "Variable px for fix imagecharges does not exist");
    if (input->variable->equalstyle(pxvar)) pxstyle = EQUAL;
    else if (input->variable->atomstyle(pxvar)) pxstyle = ATOM;
    else error->all(FLERR, "Variable px for fix imagecharges has invalid style");
  }
  if (pystr) {
    pyvar = input->variable->find(pystr);
    if (pyvar < 0) error->all(FLERR, "Variable py for fix imagecharges does not exist");
    if (input->variable->equalstyle(pyvar)) pystyle = EQUAL;
    else if (input->variable->atomstyle(pyvar)) pystyle = ATOM;
    else error->all(FLERR, "Variable py for fix imagecharges has invalid style");
  }
  if (pzstr) {
    pzvar = input->variable->find(pzstr);
    if (pzvar < 0) error->all(FLERR, "Variable pz for fix imagecharges does not exist");
    if (input->variable->equalstyle(pzvar)) pzstyle = EQUAL;
    else if (input->variable->atomstyle(pzvar)) pzstyle = ATOM;
    else error->all(FLERR, "Variable pz for fix imagecharges has invalid style");
  }
  if (nxstr) {
    nxvar = input->variable->find(nxstr);
    if (nxvar < 0) error->all(FLERR, "Variable nx for fix imagecharges does not exist");
    if (input->variable->equalstyle(nxvar)) nxstyle = EQUAL;
    else if (input->variable->atomstyle(nxvar)) nxstyle = ATOM;
    else error->all(FLERR, "Variable nx for fix imagecharges has invalid style");
  }
  if (nystr) {
    nyvar = input->variable->find(nystr);
    if (nyvar < 0) error->all(FLERR, "Variable ny for fix imagecharges does not exist");
    if (input->variable->equalstyle(nyvar)) nystyle = EQUAL;
    else if (input->variable->atomstyle(nyvar)) nystyle = ATOM;
    else error->all(FLERR, "Variable nyfor fix imagecharges has invalid style");
  }
  if (nzstr) {
    nzvar = input->variable->find(nzstr);
    if (nzvar < 0) error->all(FLERR, "Variable nz for fix imagecharges does not exist");
    if (input->variable->equalstyle(nzvar)) nzstyle = EQUAL;
    else if (input->variable->atomstyle(nzvar)) nzstyle = ATOM;
    else error->all(FLERR, "Variable nz for fix imagecharges has invalid style");
  }

  // set index and check validity of region
  if (idregion) {
    region = domain->get_region_by_id(idregion);
    if (!region) error->all(FLERR, "Region ID for fix imagecharges does not exist");
  }

  // find scale variable if set
  if (scalestr) {
    scalevar = input->variable->find(scalestr);
    if (scalevar < 0) error->all(FLERR, "Variable scale for fix imagecharges does not exist");
    if (input->variable->equalstyle(scalevar)) scalestyle = EQUAL;
    else if (input->variable->atomstyle(scalevar)) scalestyle = ATOM;
    else error->all(FLERR, "Variable scale for fix imagecharges has invalid style");
  }

  if (pxstyle == ATOM || pystyle == ATOM || pzstyle == ATOM 
      || nxstyle == ATOM || nystyle == ATOM || nzstyle == ATOM 
      || scalestyle == ATOM) varflag = ATOM;
  else if (pxstyle == EQUAL || pystyle == EQUAL || pzstyle == EQUAL 
           || nxstyle == EQUAL || nystyle == EQUAL || nzstyle == EQUAL 
           || scalestyle == EQUAL) varflag = EQUAL;
  else varflag = CONSTANT;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *type = atom->type;

  // initialize the imagei and imageid arrays
  for (int i = 0; i < nlocal + 2; i++) {
    imagei[i] = -2;
    imageid[i] = -2;
  }

  // go through and tally total charges + image types 
  int nactive = 0;
  int nimage = 0;
  int ilist[nlocal + 2]; // list to use as a mask for image charges
  for (int i = 0; i < nlocal + 2; i++) ilist[i] = 0;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (type[i] == itype) { // will become image charge
        imagei[i] = -1;
        imageid[i] = -1;
        ilist[i] = 1; // list of unassigned image charges
        nimage += 1;
      } else nactive += 1;
    }
  }

  fprintf(screen, "%d atoms of type %d to be images for %d active atoms \n", nimage, itype, nactive);

  // loop through again to assign image charges
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (type[i] != itype) { // will become image charge
        int j;
        for (j = 0; j < nlocal; j++) {
          if (ilist[j] == 1) {
            ilist[j] = 0; // mark that we're assigning this charge
            break;
          }
        }
        if (j < nlocal) { // means we found a match in loop earlier
          imagei[i] = j;
          imageid[i] = j;
        }
      }
    }
  }
}

void FixImageCharges::min_setup_pre_force(int vflag) {
  setup_pre_force(vflag);
}

void FixImageCharges::setup_pre_force(int vflag) {
  // add the image charges as new atoms
  double **x = atom->x;
  double *q = atom->q;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nadded = 0;
  int atomIndex = nlocal; // new atoms are added at index nlocal

  // update region if necessary
  if (region) region->prematch();

  if (varflag == CONSTANT) {
    for (int i = 0; i < atom->nlocal; i++)
      if (mask[i] & groupbit) {
        if (region && !region->match(x[i][0], x[i][1], x[i][2])) {
          imagei[i] = -3;
          imageid[i] = -3;
          continue;
        }
        int j = imagei[i];
        if (j != -1) { // this is not an image charge
          // transform coordinates across plane
          double nnorm = sqrt(nxvalue * nxvalue + nyvalue * nyvalue + nzvalue * nzvalue);
          double prefactor = 2 * (nxvalue / nnorm * x[i][0] + nyvalue / nnorm * x[i][1] + nzvalue / nnorm * x[i][2]);
          double delta = 2 * (nxvalue / nnorm * pxvalue + nyvalue / nnorm * pyvalue + nzvalue / nnorm * pzvalue);
          double r[3];
          r[0] = x[i][0] - (prefactor - delta) * nxvalue;
          r[1] = x[i][1] - (prefactor - delta) * nyvalue;
          r[2] = x[i][2] - (prefactor - delta) * nzvalue;
          // add a new atom
          if (j == -2) {
            nadded++;
            atom->avec->create_atom(itype, r);
            atom->q[atomIndex] = -1 * scale * q[i];
            atom->mask[atomIndex] = groupbit;
            imagei[i] = atomIndex;
            imageid[i] = atomIndex;
            imagei[atomIndex] = -1;
            imageid[atomIndex] = -1;
            atomIndex++;
          } else { // j is index of this atom's image
            atom->x[j][0] = r[0];
            atom->x[j][1] = r[1];
            atom->x[j][2] = r[2];
            atom->q[j] = -1 * scale * q[i];
          }
        }
      }
    vector_atom = imageid;
  }

  if (nadded) {
    atom->natoms += nadded;
    fprintf(screen, "fix imagecharges: added %d new atoms \n", nadded);
    if (atom->natoms < 0)
      error->all(FLERR, "Too many total atoms");
    if (atom->tag_enable) atom->tag_extend();
    if (atom->map_style) {
      atom->nghost = 0;
      atom->map_init();
      atom->map_set();
    }
  }
}

void FixImageCharges::min_pre_force(int vflag) {
  pre_force(vflag);
}

void FixImageCharges::pre_force(int vflag) {
  // move all image charges to proper locations before calculating forces
  // check to make sure all atoms in region/group have images and no extra images
  // imagei = -1 if is an image charge, -2 if not currently in region, undefined for not in group
  double ** x = atom->x;
  double * q = atom->q;
  int * mask = atom->mask;
  int nlocal = atom->nlocal;
  int nadded = 0;
  int atomIndex = nlocal;
  int seenCount = 0;
  int reqCount = 0;
  int nchanged = 0;

  int excludedHere = -1;

  bool toDelete = false;
  int dlist[nlocal + 2]; // list to use as a mask for atoms that need to be deleted
  for (int i = 0; i < nlocal + 2; i++) dlist[i] = 0;

  // update region if necessary
  if (region) region->prematch();

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      int j = imagei[i];
      if (j == -1) { // this is an image charge, will get taken care of later or deleted
        dlist[i] = !dlist[i];
        seenCount++;
      } else {
        // get new position -- transform coordinates across plane
        double nnorm = sqrt(nxvalue * nxvalue + nyvalue * nyvalue + nzvalue * nzvalue);
        double prefactor = 2 * (nxvalue / nnorm * x[i][0] + nyvalue / nnorm * x[i][1] + nzvalue / nnorm * x[i][2]);
        double delta = 2 * (nxvalue / nnorm * pxvalue + nyvalue / nnorm * pyvalue + nzvalue / nnorm * pzvalue);
        double r[3];
        r[0] = x[i][0] - (prefactor - delta) * nxvalue;
        r[1] = x[i][1] - (prefactor - delta) * nyvalue;
        r[2] = x[i][2] - (prefactor - delta) * nzvalue;
        if (j < 0 || j >= nlocal) { // used to not be in region or is new atom
          j = atomIndex;
          atomIndex++;
          nadded++;
          nchanged++;
          atom->avec->create_atom(itype, r); //add a new atom
          atom->mask[j] = groupbit;
          imagei[i] = j;
          imageid[i] = j;
          imagei[j] = -1;
          imageid[j] = -1;
        } else {
          // mark that we updated/saw its image
          dlist[j] = !dlist[j];
          reqCount++;
          // update image coordinates
          for (int k = 0; k < 3; ++k) x[j][k] = r[k];
          // update type if necessary
          if (i == exclusionAtom) {
            atom->mask[j] = groupbit;
            exclusionAtom = -1;
          }
        }
        atom->q[j] = -1 * scale * q[i]; // update charge
      }
    } else { // not in group
      int j = imagei[i];
      if (j >= 0) { // exclusion group atom
        atom->mask[j] = atom->mask[i]; // set group of image to same as atom
        atom->q[j] = 0.0; // set charge of image to zero
        dlist[j] = !dlist[j];
        reqCount++;
        exclusionAtom = i;
        excludedHere = j;
      } else if (j == -1) {
        dlist[i] = !dlist[i];
        seenCount++;
        if (i != excludedHere) mask[j] = groupbit; // just excluded image charge
      }
    }
  }
  
  // deal with the deleteList
  nlocal = atom->nlocal;
  if (seenCount > reqCount) toDelete = true;
  else if (seenCount != reqCount) error->all(FLERR, "New atom did not get image charge");

  if (toDelete) {
    int i = 0;
    while (i < atom->nlocal) {
      if (dlist[i]) {
        int endInd = atom->nlocal - 1;
        atom->avec->copy(endInd, i, 1);
        imagei[endInd] = -2;
        imageid[endInd] = -2; // zero these in case used later
        dlist[i] = dlist[endInd];
        atom->nlocal--;
        nadded--;
        nchanged++;
      } else i++;
    }
  }

  nlocal = atom->nlocal;
  for (int i = nlocal; i < nlocal * 2; ++i) { // zero out the unused parts of arrays
    if (i < atom->nmax) {
      imagei[i] = -2;
      imageid[i] = -2;
    } else error->all(FLERR, "Too many total atoms");
  }

  if (nadded) {
    atom->natoms += nadded;
    if (atom->natoms < 0) error->all(FLERR, "Too many total atoms");
    if (atom->tag_enable) atom->tag_extend();
    if (atom->map_style) {
      atom->nghost = 0;
      atom->map_init();
      atom->map_set();
    }
  }
}

void FixImageCharges::min_post_force(int vflag) {
  post_force(vflag);
}

void FixImageCharges::post_force(int vflag) {
  double ** f = atom->f;
  double ** v = atom->v;
  int nlocal = atom->nlocal;

  // update region if necessary
  if (region) region->prematch();

  if (varflag == CONSTANT) {
    for (int i = 0; i < nlocal; i++) {
      // check whether this is an image charge
      if (imagei[i] == -1) {
        f[i][0] = 0; // if so, zero forces and velocities
        f[i][1] = 0;
        f[i][2] = 0;
        v[i][0] = 0;
        v[i][1] = 0;
        v[i][2] = 0;
      }
    }
  }
}

void FixImageCharges::post_run() {
  
}

double FixImageCharges::memory_usage() {
  int nmax = atom->nmax;
  double bytes = 0.0;
  bytes += nmax * sizeof(int);
  return bytes;
}

void FixImageCharges::grow_arrays(int nmax) {
  memory->grow(this->imagei, nmax, "imagecharges::imagei");
  memory->grow(this->imageid, nmax, "imagecharges::imageid");
  vector_atom = imageid;
}

void FixImageCharges::copy_arrays(int i, int j, int delflag) {
  int i1 = imagei[i];
  imagei[j] = imagei[i];
  imageid[j] = imageid[i];
  bool found = false;

  if (i1 == -1) { // this is an image charge
    for (int x = 0; x < atom->nlocal + 1; ++x) { // need to loop over empty space at end used for sorting
      if (imagei[x] == i) {
        imagei[x] = j; //now points to new location
        imageid[x] = j;
        found = true;
        break;
      }
    }
    if (!found) fprintf(screen, "COULDN'T FIND OWNER OF IMAGECHARGE");
  }
}

void FixImageCharges::set_arrays(int i) {
  memset(&imagei[i], -2, sizeof(int));
  memset(&imageid[i], -2, sizeof(int));
}